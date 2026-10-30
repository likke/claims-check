---
name: claims-check
extends: ../../ProjectsOS/projects/design-system/CANON.md
version: beta
description: Claims Check by Dasho. A brand reviewer hands a creator video to Chicky the Judge and gets back every risky line, the proof, and a fix, then decides. Feel: warm, quick, a little cheeky, never cluttered.

colors:
  background: "#F7F6F3"   # paper, matches the logo file background
  foreground: "#171717"   # ink
  surface: "#FFFFFF"      # cards
  muted: "#5F636B"        # secondary text (AA on paper)
  border: "#E4E1DA"       # hairline
  accent: "#FEA214"       # Chicky orange (from the logo): progress, focus ring, selected state, highlights. Never body text.
  accent-foreground: "#171717"
  danger: "#F72422"       # marker red (from the logo): strike-through, fail chips, MISSING stamp
  danger-soft: "#FDE8E8"
  success: "#1F8A5B"      # pass
  success-soft: "#E3F4EC"
  accent-soft: "#FFF1D6"

typography:
  display:   { fontFamily: "Poppins", fontSize: "48px", fontWeight: 700, lineHeight: 1.1, letterSpacing: "-0.01em" }   # mobile 34px
  h2:        { fontFamily: "Poppins", fontSize: "34px", fontWeight: 700, lineHeight: 1.15 }
  h3:        { fontFamily: "Poppins", fontSize: "26px", fontWeight: 600, lineHeight: 1.2 }
  lead:      { fontFamily: "Inter", fontSize: "20px", fontWeight: 400, lineHeight: 1.5 }
  body-md:   { fontFamily: "Inter", fontSize: "17px", fontWeight: 400, lineHeight: 1.55 }
  caption:   { fontFamily: "Inter", fontSize: "14px", fontWeight: 500, lineHeight: 1.4 }
  # Exactly six sizes: 14, 17, 20, 26, 34, 48. No uppercase eyebrows. No serif.

spacing: { xs: "8px", sm: "16px", md: "24px", lg: "32px", xl: "48px", 2xl: "64px" }
radii: { card: "20px", input: "14px", pill: "999px" }
---

## The user's job (write every screen for this)
Who: a brand manager or agency account lead in the Philippines, at a laptop at work (sometimes on a phone), with a creator's video that must post today.
What they want: know in two minutes whether the video is safe to post, which lines are risky, the proof for each, and send fixes back.
Order: 1 start (sample, upload, or replay) -> 2 watch Chicky check -> 3 verdict -> 4 go line by line (claim, reason, proof, fix) -> 5 disclosure -> 6 decide (request changes or approve) -> 7 done.

## Principles
1. One job per screen, one primary button per screen, placed under the content it acts on.
2. Chicky is the host and the one memorable thing: big (160px desktop, 112px mobile), a speech bubble with a tail, one reaction per screen. Everything else stays quiet.
3. The marker-red strike is the brand move: struck original line in red on red-soft, fix in ink on success-soft.
4. Centred single column (max 720px) for every flow screen except the claim screen (video 7/12 left, card 5/12 right; stacks on mobile). Text left-aligned inside the column.
5. Buttons: primary = ink pill 56px; secondary = white pill with 1.5px ink border; tertiary = icon + text, no underline. No underlined action links anywhere.
6. No box-shadow, no coloured side rails, no gradients, no emoji, no uppercase eyebrow labels, no middle-dot meta strings, no arrows appended to button text.
7. Motion: one moment per screen (Chicky pop on reaction, stamp slam, confetti on done). 150 to 300ms, ease-out, reduced-motion respected.
8. Body text 17px minimum. Tap targets 44px minimum. Focus ring 3px Chicky orange.
