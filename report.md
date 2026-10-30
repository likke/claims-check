# Claims Check UI v3 — component change report

1. **Buttons** — Before: mixed underlined quiet links, inconsistent controls, and a text-only “New check”. After: consistent 52px ink primary pills, outlined secondary pills, icon-led tertiary buttons, visible focus treatment, and a small “New check” pill.
2. **Welcome** — Before: secondary choices were tiny links and the intro action was plain text. After: one clear sample CTA, three icon option tiles, a video play chip, and a tertiary intro skip control. Kept the AI label and byline.
3. **Sample picker** — Before: unselected flat rows with weak state. After: larger radio-style cards with selected orange border/tint and a check mark.
4. **Upload** — Before: raw file inputs and basic fields. After: dashed video/image drop zones, hidden file inputs with selected names, labelled rounded form controls, and a full-width submit action on mobile.
5. **Sisiw** — Before: small mascot and flat speech panels. After: 140px desktop and 96px mobile mascot sizing, bob/pop motion, white speech panels with tails, and reduced-motion handling.
6. **Sound toggle** — Before: “Sound on/off” text. After: round 40px button with speaker and speaker-off SVG states, aria-pressed, accessible label, and visually hidden text.
7. **Checking screen** — Before: plain step text and no progress indicator. After: vertical spinner/check stepper and orange progress bar below the video, with step state updated from review events.
8. **Results** — Before: muted counts and small reaction. After: 48px verdict line, red fix and green pass chips, and a larger Sisiw reaction.
9. **Claim screen** — Before: plain claim text, underlined source, and small timecode. After: red struck original, green suggestion panel, inline source card toggle, and pill timecode seek control.
10. **Disclosure** — Before: plain disclosure copy and text replay link. After: status card with PASSED/MISSING stamp and icon-led replay button.
11. **Decision** — Before: uneven links/buttons. After: named reviewer field and two large side-by-side actions with clear disabled styling; changes is primary when fixes are needed, approval is disabled until allowed.
12. **Done** — Before: plain completion view. After: celebratory Sisiw frame, 1s CSS confetti burst, reduced-motion support, and primary “Check another post”.
13. **Details/event log** — Before: raw browser disclosure triangle and a generic “Details” action. After: styled “Show what the agent did” pill opens the existing event-log disclosure card.

Files changed: `static/review.html`, `static/design-system.css`.
