/* Pure timeline and copy helpers shared by the reviewer and its tests. */
(() => {
  function activeReactionAt(notes, time) {
    if (!Number.isFinite(time)) return null;
    return notes
      .filter(note => Number.isFinite(note.time) && time >= note.time && time < note.time + 4)
      .sort((a, b) => b.time - a.time || b.n - a.n)[0] || null;
  }

  function fallbackReaction(note) {
    if (note.reaction) return note.reaction;
    if (note.kind === 'disclosure') return 'Nice, the partnership label is right there!';
    if (note.kind === 'nodisclosure') return 'Let’s find a clear partnership label.';
    if (note.kind === 'message') return note.message?.status === 'covered' ? 'That key point came through!' : 'Could we bring this point in?';
    if (note.claim?.review) return 'I’m not sure about this language. Let’s check together.';
    if (note.kind === 'verified') return 'That claim lines up with the sources!';
    const first = note.claim?.rounds?.[0] || {};
    if (first.judgement === 'rule') return 'This one needs a closer look.';
    if (first.judgement === 'contradicted' || first.verdict === 'fail') return 'Wait, let’s check this claim closely!';
    return 'I need a little more support here.';
  }

  window.ReviewReactions = {activeReactionAt, fallbackReaction};
})();
