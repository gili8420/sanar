# Merge protocol

The rules for turning N transcription passes into one final transcript.

## The contract

**You are a scribe, not an editor.** The deliverable is what was said, as it was said.

You are permitted to fix exactly one class of error: mistakes made by the microphone and
the model. You are forbidden from fixing mistakes made by the speaker.

| Error | Action |
|---|---|
| Whisper heard "кубернетис в кластере" for "Kubernetes в кластере" | Fix — model error |
| Whisper wrote "Кафка" where the glossary says "Kafka" | Fix — model error |
| Speaker says "Postgres — это NoSQL" | **Keep** — speaker's claim, not yours |
| Speaker says "мы задеплоили в пятницу" but the doc says Thursday | **Keep** the speaker |
| Speaker's sentence is ungrammatical, unfinished, or self-contradicting | **Keep** as-is |
| Speaker repeats themselves three times | **Keep** all three |
| Speaker uses the wrong term for their own system | **Keep** — note it in merge-notes |
| One pass invented a sentence the others don't have | **Drop** — hallucination |

If you are unsure which class an error belongs to, it belongs to the speaker. Keep it.

## Why this rule is absolute

A transcript is evidence. Its value is that it is not your opinion. A summarized,
smoothed, fact-checked transcript is worth less than no transcript, because it looks
authoritative while having silently lost the thing it was made to preserve — and nobody can
tell which parts you changed. An objectively wrong statement, preserved exactly, is a
correct transcript.

You will feel pressure to help: to finish the clipped sentence, to fix the misused term, to
drop the third repetition, to make the argument read better. That pressure is the failure
mode. Resist it every single time.

## Procedure

### 1. Align

Segment timestamps are the join key. Passes segment differently, so align by overlapping
time spans, not by line number. Work through the timeline once, front to back, in windows
of a few segments.

Use `audio.chapters.txt` to insert `## <source file>` headers at the right offsets.

### 2. Per window, choose

For each span, you have one candidate per pass. Pick using this order:

1. **Glossary consistency.** A candidate that spells a glossary term correctly beats one
   that mangles it. This is the main reason multiple passes exist.
2. **Code-switch fidelity.** In mixed speech, the `en` pass usually gets Latin terms right
   and the surrounding Russian wrong; the `ru` pass does the opposite. Take the terms from
   the `en` pass and the connective speech from the `ru` pass, **at word granularity**. This
   is the highest-value operation in the whole merge.
3. **Phonetic plausibility.** Which candidate could have produced the same sound? A
   candidate that is semantically nicer but phonetically distant is a hallucination.
4. **Corroboration.** Where passes agree, that text is settled. Where only one pass has
   content, suspect it.
5. **Coherence — last, and weakly.** Never let coherence override 1-4. "It reads better"
   is not evidence.

Splicing words across passes is expected and correct. Inventing words that appear in no
pass is not, ever.

### 3. Strip ASR artifacts

Remove only what is unambiguously not speech:

- Decoder loops: the same phrase repeated far past plausibility. Keep one instance, note
  the span in merge-notes.
- Training-data bleed: subtitle credits, "Спасибо за просмотр", "Thanks for watching",
  "Продолжение следует", channel names, "Субтитры сделал ...", ad reads — when the audio
  clearly has no such speech.
- Text generated over silence or music (a common VAD-gap artifact).

Everything else stays, including filler, false starts, and stutters that the passes
captured. Do not add filler that no pass captured.

### 4. Mark uncertainty, don't resolve it silently

When two candidates are both plausible and the choice changes the meaning:

```
[10:23] ... и там у нас [?шардирование|шифрование] по клиентам
```

When a span is unintelligible in every pass:

```
[10:31] ... это работает через [unintelligible ~3s] и дальше в очередь
```

An honest `[?]` is worth more than a confident guess. Never delete a span you could not
resolve — a gap in the transcript is invisible; a marked gap is a question someone can
answer.

### 5. Output

`transcript.md`:

- `## <source file>` per stitched input, at its chapter offset.
- `[hh:mm:ss] text` lines, wrapped to the segment structure you merged.
- Speaker labels only if the passes actually support them (`**S1:**`); never invent
  speakers or attribute lines by guessing.
- A short front-matter block: total duration, source files, languages, model, glossary size.
- Nothing else. No summary, no key points, no action items, no editorial notes in the body.
  If the user wants a summary, that is a separate ask, in a separate file.

`merge-notes.md`:

- Every span where passes disagreed materially: timestamp, each candidate, the choice, the
  reason (which rule above).
- Every artifact stripped, with its span.
- Every `[?]` and `[unintelligible]` left in.
- Glossary terms that appeared in the audio but were dropped from the prompt (from the
  pass sidecar JSON) — these are where residual errors concentrate.
- Places where the speaker contradicted the documents. **Listed here, never fixed in the
  transcript.**

This file is what makes the transcript auditable. Someone must be able to reconstruct every
decision you made without re-listening.

## Self-check before delivering

- Does every sentence in `transcript.md` trace back to at least one pass? Find one that
  doesn't — that is your hallucination.
- Did you fix any statement because it was *wrong* rather than *misheard*? Revert it.
- Is there a summary, conclusion, or "key takeaway" anywhere in the body? Delete it.
- Is the transcript materially shorter than the longest pass? You dropped content. Find it.
- Does every glossary term appear with one consistent spelling throughout?
