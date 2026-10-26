"""Dasho Claim Check: source-grounded review of creator video claims."""
import json, os, re, shutil, subprocess, tempfile, uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
FIXTURE_SOURCE = "Ledgerly AI helps teams organize invoices and flag unusual charges for human review. It does not guarantee fraud prevention or regulatory compliance."
TOOLS = ["search_source", "fetch_source", "judge_claim", "rewrite_line", "check_disclosure", "check_coverage", "finish"]

def extract_media(video: bytes) -> tuple[bytes, list[dict[str, Any]]]:
    """Extract mono 16 kHz audio and up to six evenly spaced one-second frames."""
    with tempfile.TemporaryDirectory(prefix="claim-check-") as tmp:
        source = Path(tmp) / "upload.bin"
        audio_path = Path(tmp) / "audio.wav"
        source.write_bytes(video)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)], check=True, capture_output=True)
        duration_text = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(source)], check=True, capture_output=True, text=True).stdout.strip()
        duration = max(1, int(float(duration_text) + .5))
        count = min(6, max(3, duration))
        times = sorted(set(round(i * max(0, duration - 1) / max(1, count - 1), 2) for i in range(count)))
        frames = []
        for i, at in enumerate(times):
            frame = Path(tmp) / f"frame-{i}.jpg"
            subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", str(source), "-frames:v", "1", "-q:v", "3", str(frame)], check=True, capture_output=True)
            frames.append({"time": at, "image": frame.read_bytes()})
        return audio_path.read_bytes(), frames

class FixtureProvider:
    def transcribe(self, video: bytes) -> str: return "Ledgerly AI guarantees zero FDA violations. Sulit ito!"
    def frames(self, video: bytes) -> list[dict[str, Any]]: return [{"time": 1.2, "text": "Paid partnership", "claims": []}]
    def plan(self, context, history): return {"tool": "search_source", "input": {"query": "capabilities"}, "mode": "fixture"}
    def json_plan(self, prompt): return {"tool": "finish", "input": {}, "mode": "json"}
    def judge_claim(self, claim, evidence): return judge_claim(claim, evidence)
    def rewrite_line(self, claim, reason, evidence, round_number=1): return rewrite_line(claim, reason, evidence)
    def check_disclosure(self, transcript, frames): return check_disclosure(transcript, frames)
    def check_coverage(self, transcript, key_messages): return check_coverage(transcript, key_messages)

class MetaProvider(FixtureProvider):
    """Meta.ai v1 provider for Muse Voice Transcribe and Muse Spark."""
    def __init__(self):
        self.base_url = os.getenv("MODEL_BASE_URL", "https://api.meta.ai/v1").rstrip("/")
        self.api_key = os.environ["MODEL_API_KEY"]
        self.asr_model = os.getenv("ASR_MODEL", "muse-voice-transcribe")
        self.model = os.getenv("MODEL_NAME", "muse-spark")
        self.mode = "tools"

    def _request(self, path, payload):
        import urllib.error, urllib.request
        req = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json", "Authorization":f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as res: return json.loads(res.read())

    def transcribe(self, video):
        audio, raw_frames = extract_media(video)
        import base64
        result = self._request("/audio/transcriptions", {"model":self.asr_model, "file":base64.b64encode(audio).decode(), "language":"tl,en", "prompt":"Tagalog and English (Taglish) speech"})
        self._frames = raw_frames
        return result.get("text", result.get("transcript", ""))

    def frames(self, video):
        import base64
        return [{"time":f["time"], "image":base64.b64encode(f["image"]).decode()} for f in getattr(self, "_frames", [])]

    def _spark(self, instruction, images=None):
        content = [{"type":"input_text", "text":instruction}]
        for image in images or []: content.append({"type":"input_image", "image_url":"data:image/jpeg;base64,"+image["image"]})
        result = self._request("/responses", {"model":self.model, "input":content})
        return result

    @staticmethod
    def _output(result):
        if "output_text" in result: return result["output_text"]
        for item in result.get("output", []):
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"): return c.get("text", "")
        return ""

    def _json_call(self, prompt, images=None):
        raw = self._output(self._spark(prompt, images))
        for attempt in range(2):
            try:
                match = re.search(r"\{.*\}", raw, re.S)
                return json.loads(match.group() if match else raw)
            except (TypeError, json.JSONDecodeError):
                if attempt: raise
                raw = self._output(self._spark("Return valid JSON only. Previous response was invalid.\n" + prompt, images))

    def plan(self, context, history):
        schema = [{"type":"function", "name":name, "description":f"Run {name}", "parameters":{"type":"object", "properties":{"query":{"type":"string"},"claim":{"type":"string"}}, "additionalProperties":True}} for name in TOOLS]
        try:
            result = self._request("/responses", {"model":self.model, "input":[{"type":"input_text","text":json.dumps({"context":context,"history":history})}], "tools":schema})
            for item in result.get("output", []):
                if item.get("type") == "function_call":
                    self.mode="tools"
                    return {"tool":item.get("name"), "input":json.loads(item.get("arguments", "{}")), "mode":"tools"}
            raise ValueError("No tool call in response")
        except Exception as error:
            # Any HTTP 4xx or incompatible tool response uses the JSON planner.
            self.mode="json"
            plan=self._json_call("Choose one tool from " + json.dumps(TOOLS) + ". Return {tool,input}. Context/history: " + json.dumps({"context":context,"history":history}))
            plan["mode"]="json"
            return plan

    def json_plan(self, prompt): return self._json_call(prompt)
    def judge_claim(self, claim, evidence):
        prompt=f"Judge whether this advertising claim is supported by sources. Return JSON verdict (pass/fail), evidence, reason. Claim: {claim}\nSources: {evidence}"
        return self._json_call(prompt)
    def rewrite_line(self, claim, reason, evidence, round_number=1):
        return self._output(self._spark(f"Rewrite this line to be true according to sources. Preserve the creator's point and voice; never delete the claim. Round {round_number}. Failure reason: {reason}\nLine: {claim}\nSources: {evidence}"))
    def check_disclosure(self, transcript, frames):
        return bool(self._json_call("Return JSON {disclosed:boolean}. Is paid disclosure present in the speech or these video frames? Speech: " + transcript, frames).get("disclosed"))
    def check_coverage(self, transcript, key_messages):
        return self._json_call(f"Return JSON covered and missing arrays. Speech: {transcript}\nRequired messages: {key_messages}")

def get_provider(): return MetaProvider() if os.getenv("MODEL_API_KEY") else FixtureProvider()
def extract_claims(transcript):
    return [{"quote":m.group().strip(),"timestamp":round(m.start()/12,1),"rounds":[]} for m in re.finditer(r"[^.!?]+[.!?]?", transcript) if re.search(r"\b(guarantee|guarantees|cure|treat|prevents|eliminates|100%|zero)\b",m.group(),re.I)]
def search_source(query, sources): return [line.strip() for line in sources.splitlines() if any(w.lower() in line.lower() for w in re.findall(r"\w+",query) if len(w)>3)] or [sources.strip()]
def judge_claim(claim,evidence):
    supported=bool(evidence) and not re.search(r"guarantee|zero FDA|cure|treat",claim,re.I)
    return {"verdict":"pass" if supported else "fail","evidence":evidence[0] if evidence else "No supporting source found.","reason":"The wording is supported by the source." if supported else "The source describes assistance and human review, not a guarantee or therapeutic outcome."}
def rewrite_line(claim,reason,evidence): return "Ledgerly AI helps flag possible issues early, so your team can review them before posting. Sulit ito!"
def check_disclosure(transcript,frames): return bool(re.search(r"paid partnership|sponsored|ad\b|pakikipagtulungan bayad|bayad na partnership",transcript,re.I) or any(re.search(r"paid partnership|sponsored|\bad\b",f.get("text",""),re.I) for f in frames))
def check_coverage(transcript,key_messages):
    messages=[x.strip() for x in key_messages.splitlines() if x.strip()]
    covered=[m for m in messages if any(w.lower() in transcript.lower() for w in re.findall(r"\w+",m) if len(w)>4)]
    return {"covered":covered,"missing":[m for m in messages if m not in covered]}

def run_review(video,sources,messages,pack,provider=None,emit=None):
    provider=provider or get_provider(); trace=[]
    def log(tool,inputs,output,mode=None):
        if len(trace)<12:
            step={"step":len(trace)+1,"tool":tool,"input":inputs,"output":output}
            if mode: step["mode"]=mode
            trace.append(step)
            if emit: emit(step)
    transcript=provider.transcribe(video); log("transcribe",{"video_bytes":len(video)},transcript)
    frames=provider.frames(video); log("vision_frames",{"sampled_frames":len(frames)},[{k:v for k,v in f.items() if k!="image"} for f in frames])
    claims=extract_claims(transcript); log("extract_claims",{},claims)
    for claim in claims:
        evidence=search_source(claim["quote"],sources); log("search_source",{"query":claim["quote"]},evidence)
        verdict=provider.judge_claim(claim["quote"],evidence); log("judge_claim",{"claim":claim["quote"]},verdict); claim["rounds"].append(verdict)
        if verdict.get("verdict")=="fail":
            revised=provider.rewrite_line(claim["quote"],verdict.get("reason",""),evidence,1); log("rewrite_line",{"claim":claim["quote"],"failure_reason":verdict.get("reason")},revised)
            retry=provider.judge_claim(revised,evidence); log("judge_claim",{"claim":revised,"round":2,"failure_reason":verdict.get("reason")},retry); claim["rounds"].append(retry); claim["suggested_line"]=revised
        claim["verdict"]=claim["rounds"][-1].get("verdict","fail"); claim["evidence"]=claim["rounds"][-1].get("evidence","")
    disclosed=provider.check_disclosure(transcript,frames); log("check_disclosure",{},disclosed)
    coverage=provider.check_coverage(transcript,messages); log("check_coverage",{},coverage)
    log("finish",{}, {"claim_count":len(claims)})
    return {"id":str(uuid.uuid4()),"transcript":transcript,"claims":claims,"disclosure":disclosed,"coverage":coverage,"trace":trace,"frames":[{k:v for k,v in f.items() if k!="image"} for f in frames]}

app=FastAPI(title="Dasho Claim Check"); app.mount("/static",StaticFiles(directory=ROOT/"static"),name="static"); REVIEWS={}
@app.get("/",response_class=HTMLResponse)
def home(): return (ROOT/"static"/"index.html").read_text()
@app.post("/api/review")
async def review(video:UploadFile=File(...),sources:str=Form(FIXTURE_SOURCE),key_messages:str=Form("Organize invoices\nFlag unusual charges"),pack:str=Form("Tech claims")):
    result=run_review(await video.read(),sources,key_messages,pack); REVIEWS[result["id"]]=result; return result
@app.get("/api/review/{review_id}/trace")
def trace(review_id):
    result=REVIEWS.get(review_id)
    if not result:return {"error":"Review not found"}
    return {"trace":result["trace"]}
@app.get("/api/review/{review_id}/events")
async def events(review_id):
    async def stream():
        for item in REVIEWS.get(review_id,{}).get("trace",[]): yield "data: "+json.dumps(item)+"\n\n"
    return StreamingResponse(stream(),media_type="text/event-stream")
@app.post("/api/review/{review_id}/decision")
async def decision(review_id,payload:dict):
    if review_id not in REVIEWS:return {"error":"Review not found"}
    REVIEWS[review_id]["decision"]={"status":payload.get("status"),"reviewer":payload.get("reviewer","")}; return REVIEWS[review_id]["decision"]
