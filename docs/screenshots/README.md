# Screenshots

Placeholder directory for the images referenced in the main
[README](../../README.md). Drop real PNG captures here with these exact names so
the README renders them:

| File | What to capture |
| --- | --- |
| `multi-doc-query.png` | An answer to a cross-cutting question, showing citations from **multiple files** and the *"Answer drawn from N documents: …"* attribution line. |
| `citations.png` | An answer with clickable `[n]` badges and the **Sources** panel — each card showing filename, page/locator, the confidence bar, and the full passage (ideally with a sentence highlighted). |
| `insufficient-context.png` | The honesty state: ask an off-topic question so the amber *"The documents don't clearly answer this — here's the closest related material"* banner appears with the weak sources below. |
| `chat-thread.png` | A multi-turn conversation where a follow-up was rewritten — show the *"Interpreted as: …"* line and at least two stacked turns. |

To capture: run the backend (`uvicorn app.main:app --reload`) and frontend
(`npm run dev`) with a chat model configured, index two or three small documents
with overlapping topics, and screenshot each scenario above at a desktop width.
