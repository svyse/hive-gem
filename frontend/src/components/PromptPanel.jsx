import React, { useEffect, useRef, useState } from 'react'
import { formatSpeech, listSpeechReplacements, submitSpeechFeedback } from '../api/client.js'

const QA_ORCHESTRATORS = [
  { value: 'interactive', label: 'Interactive STEM (Science/Tech/Engineering/Math/Biology)' },
  { value: 'hr', label: 'HR orchestrator' },
  { value: 'law', label: 'Law (router)' },
  { value: 'law_india', label: 'Law (India)' },
  { value: 'law_international', label: 'Law (International)' },
  { value: 'finance', label: 'Finance' },
  { value: 'economics', label: 'Economics' },
  { value: 'social', label: 'Social dynamics & speech' },
  { value: 'medicine', label: 'Medicine' },
  { value: 'web_design', label: 'Web design (UI/UX, HTML/CSS, accessibility)' },
  { value: 'computer_vision', label: 'Computer vision' },
  { value: 'cyber', label: 'Cybersecurity' }
]

export default function PromptPanel({
  mode,
  setMode,
  qaOrchestrator,
  setQaOrchestrator,
  inputMode,
  setInputMode,

  projectPath,
  setProjectPath,
  prompt,
  setPrompt,
  copyToWorkspace,
  setCopyToWorkspace,
  speakVoiceResponses,
  setSpeakVoiceResponses,
  llmBackend = 'local',
  llmStatus = null,
  modelSwitching = false,
  modelSwitchError = null,
  onLlmBackendChange,
  onStart,
  disabled
}) {
  const [listening, setListening] = useState(false)
  const recognitionRef = useRef(null)
  const shouldListenRef = useRef(false)
  const restartBackoffMsRef = useRef(200)

  // When enabled: after you press 🛑 Stop, automatically submit the current prompt.
  const [autoSubmitVoice, setAutoSubmitVoice] = useState(true)
  const autoSubmitVoiceRef = useRef(true)

  const [speechLang, setSpeechLang] = useState(() => {
    try {
      return window.localStorage.getItem('speechRecognitionLang') || 'en-IN'
    } catch (_) {
      return 'en-IN'
    }
  })
  const speechLangRef = useRef(speechLang)

  // Keep refs for latest values (used inside timeouts / event handlers).
  const modeRef = useRef(mode)
  const promptRef = useRef(prompt)
  const projectPathRef = useRef(projectPath)

  // List mode is used only when explicitly triggered (e.g. "following points").
  // To avoid "everything becomes bullets", it auto-expires after a few items.
  const listModeRef = useRef({ active: false, itemsLeft: 0, kind: 'bullet' })

  // For learning: capture a single dictation session's raw + formatted text.
  const speechSessionRawRef = useRef('')
  const speechSessionFormattedRef = useRef('')

  // For "learn from corrections": after dictation ends, if the user edits the dictation text,
  // we send raw/formatted/final to the backend so replacements can be learned.
  const insertingFromSpeechRef = useRef(false)
  const pendingFinalizeFeedbackRef = useRef(false)
  const speechFeedbackRef = useRef(null) // {mode, raw, formatted, prefix, createdAt}
  const feedbackTimerRef = useRef(null)

  const [speechReplacements, setSpeechReplacements] = useState([])
  const replacementsRef = useRef([])

  useEffect(() => {
    modeRef.current = mode
  }, [mode])

  useEffect(() => {
    promptRef.current = prompt
  }, [prompt])

  useEffect(() => {
    projectPathRef.current = projectPath
  }, [projectPath])

  useEffect(() => {
    autoSubmitVoiceRef.current = !!autoSubmitVoice
  }, [autoSubmitVoice])

  useEffect(() => {
    speechLangRef.current = speechLang || 'en-IN'
    try {
      window.localStorage.setItem('speechRecognitionLang', speechLangRef.current)
      if (recognitionRef.current) recognitionRef.current.lang = speechLangRef.current
    } catch (_) {}
  }, [speechLang])

  useEffect(() => {
    replacementsRef.current = speechReplacements || []
  }, [speechReplacements])

  useEffect(() => {
    let cancelled = false
    const m = mode === 'code' ? 'code' : 'qa'
    listSpeechReplacements({ mode: m, limit: 120 })
      .then((res) => {
        if (cancelled) return
        setSpeechReplacements(res?.items || [])
      })
      .catch(() => {
        // ok: backend might not have speech endpoints yet
      })
    return () => {
      cancelled = true
    }
  }, [mode])

  function normalizeDictation(text) {
    if (!text) return ''
    let t = String(text)

    // Spoken punctuation tokens (conservative)
    const repls = [
      [/\bcomma\b/gi, ','],
      [/\b(full\s+stop|period)\b/gi, '.'],
      [/\bquestion\s+mark\b/gi, '?'],
      [/\bexclamation\s+mark\b/gi, '!'],
      [/\bnew\s+line\b/gi, '\n'],
      [/\bunderscore\b/gi, '_'],
      [/\b(dash|hyphen)\b/gi, '-'],
      [/\bat\s+sign\b/gi, '@'],
      [/\bsample\s+projects\b/gi, 'sample_projects'],
      [/\bnode\s+modules\b/gi, 'node_modules']
    ]
    for (const [rx, rep] of repls) t = t.replace(rx, rep)

    if (modeRef.current === 'code') {
      const codeRepls = [
        [/\bopen\s+parenthesis\b/gi, '('],
        [/\b(close|closed)\s+parenthesis\b/gi, ')'],
        [/\bopen\s+bracket\b/gi, '['],
        [/\b(close|closed)\s+bracket\b/gi, ']'],
        [/\bopen\s+(curly\s+)?brace\b/gi, '{'],
        [/\b(close|closed)\s+(curly\s+)?brace\b/gi, '}'],
        [/\bcolon\b/gi, ':'],
        [/\bsemicolon\b/gi, ';'],
        [/\bequals?(\s+sign)?\b/gi, '='],
        [/\bforward\s+slash\b/gi, '/'],
        [/\bback\s+slash\b/gi, '\\'],
        [/\bdouble\s+quote\b/gi, '"'],
        [/\bsingle\s+quote\b/gi, "'"]
      ]
      for (const [rx, rep] of codeRepls) t = t.replace(rx, rep)
    }

    // "dot com" / "dot py" etc (conservative)
    t = t.replace(/\bdot\s+(com|net|org|io|ai|py|js|jsx|ts|tsx|json|html|css|md|txt|env|yaml|yml)\b/gi, '.$1')

    // Merge spelled letters like "g p s" -> "gps"
    t = t.replace(/(?:\b[A-Za-z]\b(?:\s+|$)){3,}/g, (m) => {
      const letters = m.match(/[A-Za-z]/g) || []
      return letters.join('')
    })

    // Fix weird artifacts like ".2"
    t = t.replace(/([.!?])\s*\d+\b/g, '$1')

    // Collapse spaces
    t = t.replace(/[ \t]+/g, ' ')
    t = t.replace(/ *\n */g, '\n')
    return t.trim()
  }

  function applyReplacements(text) {
    const reps = replacementsRef.current || []
    if (!text || !reps.length) return text
    let out = text

    // Apply longer src first
    const sorted = [...reps].sort((a, b) => (b?.src?.length || 0) - (a?.src?.length || 0))
    for (const r of sorted.slice(0, 120)) {
      const src = (r?.src || '').trim()
      const dst = (r?.dst || '').trim()
      if (!src || !dst) continue
      if (/^[A-Za-z0-9_]+$/.test(src)) {
        out = out.replace(new RegExp(`\\b${src.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi'), dst)
      } else {
        out = out.replace(new RegExp(src.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), dst)
      }
    }
    return out
  }

  function splitIntroAndPoints(text) {
    const t = (text || '').trim()
    if (!t) return { intro: '', points: [] }

    // Markers like: "first point", "second point", "point number one", "point two"
    const marker1 =
      /\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\s*(?:point|bullet)\b/gi
    const marker2 = /\bpoint\s*(?:number\s*)?(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b/gi

    const sep = '\n<<<PT>>>\n'
    let t2 = t.replace(marker1, sep).replace(marker2, sep)
    if (!t2.includes('<<<PT>>>')) return { intro: t, points: [] }

    const parts = t2
      .split('<<<PT>>>')
      .map((p) => p.replace(/^\s*[-:]+\s*/g, '').trim())
      .filter(Boolean)

    const intro = parts[0] || ''
    const points = parts.slice(1)
    return { intro, points }
  }


  function splitIntroAndOptions(text) {
    const t = (text || '').trim()
    if (!t) return { intro: '', options: [] }

    const toLabel = (raw) => {
      const s = String(raw || '').trim()
      const low = s.toLowerCase()
      const map = { one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9', ten: '10' }
      if (map[low]) return map[low]
      if (/^[a-z]$/i.test(s)) return s.toUpperCase()
      if (/^\d+$/.test(s)) return String(parseInt(s, 10))
      return s.toUpperCase()
    }

    const sep = '\n<<<OPT:$1>>>\n'
    const marker = /\b(?:option|choice)\s*(?:number\s*)?([a-j]|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b/gi
    const t2 = t.replace(marker, (_m, g1) => `\n<<<OPT:${toLabel(g1)}>>>\n`)
    if (!t2.includes('<<<OPT:')) return { intro: t, options: [] }

    const [introPart, ...rest] = t2.split('<<<OPT:')
    const intro = introPart.replace(/^\s*[-:]+\s*/g, '').trim()
    const options = []
    for (const part of rest) {
      const idx = part.indexOf('>>>')
      if (idx === -1) continue
      const label = part.slice(0, idx).trim()
      const body = part.slice(idx + 3).replace(/^\s*[-:]+\s*/g, '').trim()
      if (body) options.push({ label, body })
    }
    return { intro, options }
  }

  function smartFormatSegment(segment) {
    let text = normalizeDictation(segment)
    text = applyReplacements(text)

    if (!text) return { out: '', listModeChanged: false, structured: false }

    // Explicitly end list mode
    if (/\b(?:end|stop)\s+(?:list|points|bullets|options|choices)\b/i.test(text)) {
      listModeRef.current = { active: false, itemsLeft: 0, kind: 'bullet' }
      const cleaned = text.replace(/\b(?:end|stop)\s+(?:list|points|bullets|options|choices)\b/gi, '').trim()
      return { out: cleaned, listModeChanged: true, structured: false }
    }

    let triggeredList = false
    if (/\b(?:the\s+)?following\s+(?:bullet\s+)?points\b/i.test(text)) {
      triggeredList = true
      listModeRef.current = { active: true, itemsLeft: 6, kind: 'bullet' }
      text = text.replace(/\b(?:the\s+)?following\s+(?:bullet\s+)?points\b/gi, '').trim()
      if (!text) return { out: '', listModeChanged: true, structured: true }
    }

    let triggeredOptions = false
    if (/\b(?:the\s+)?(?:following\s+)?(?:answer\s+)?(?:choices?|options?)\b/i.test(text)) {
      triggeredOptions = true
      listModeRef.current = { active: true, itemsLeft: 8, kind: 'option' }
      text = text.replace(/\b(?:the\s+)?(?:following\s+)?(?:answer\s+)?(?:choices?|options?)\b/gi, '').trim()
      if (!text) return { out: '', listModeChanged: true, structured: true }
    }

    const { intro: optIntro, options } = splitIntroAndOptions(text)
    if (options.length) {
      const lines = []
      if (optIntro) lines.push(optIntro)
      for (const opt of options) lines.push(`- ${opt.label}. ${opt.body}`)
      return { out: lines.join('\n').trim(), listModeChanged: triggeredOptions, structured: true }
    }

    const { intro, points } = splitIntroAndPoints(text)
    const lines = []

    if (points.length) {
      if (intro) lines.push(intro)
      for (const p of points) {
        if (p) lines.push(`- ${p}`)
      }
      return { out: lines.join('\n').trim(), listModeChanged: triggeredList, structured: true }
    }

    const lm = listModeRef.current
    if (triggeredList || triggeredOptions || lm.active) {
      const wordCount = (text.match(/\w+/g) || []).length
      if (!triggeredList && !triggeredOptions && wordCount > 24) {
        listModeRef.current = { active: false, itemsLeft: 0, kind: 'bullet' }
        return { out: text, listModeChanged: true, structured: false }
      }

      if (!triggeredList && !triggeredOptions && lm.active && lm.itemsLeft > 0) {
        lm.itemsLeft -= 1
        if (lm.itemsLeft <= 0) {
          listModeRef.current = { active: false, itemsLeft: 0, kind: 'bullet' }
        }
      }

      return { out: `- ${text}`.trim(), listModeChanged: triggeredList || triggeredOptions, structured: true }
    }

    return { out: text, listModeChanged: false, structured: false }
  }

  function bestTranscriptFromResult(res) {
    if (!res) return ''
    let best = res?.[0] || null
    try {
      for (let i = 0; i < res.length; i++) {
        const alt = res[i]
        if (!alt?.transcript) continue
        const curConf = Number.isFinite(alt.confidence) ? alt.confidence : 0
        const bestConf = best && Number.isFinite(best.confidence) ? best.confidence : 0
        if (!best || curConf > bestConf) best = alt
      }
    } catch (_) {}
    return String(best?.transcript || '').trim()
  }

  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) return

    const r = new SR()
    r.continuous = true
    r.interimResults = true
    r.lang = speechLangRef.current || 'en-IN'
    try {
      r.maxAlternatives = 3
    } catch (_) {}

    r.onresult = (event) => {
      ;(async () => {
        try {
          const finals = []
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const res = event.results[i]
            const transcript = bestTranscriptFromResult(res)
            if (!transcript) continue
            if (res.isFinal) finals.push(transcript)
          }

          if (!finals.length) return

          const formattedPieces = []
          const fmtMode = modeRef.current === 'code' ? 'code' : 'qa'
          for (const seg of finals) {
            const segRaw = normalizeDictation(seg)
            let out = ''
            const localSmart = smartFormatSegment(segRaw)
            try {
              const remote = await formatSpeech({ text: segRaw, mode: fmtMode })
              out = String(remote?.formatted_text || '').trim()
              if ((!out || localSmart.structured) && localSmart.out) {
                out = String(localSmart.out || '').trim()
              }
            } catch (_) {
              out = String(localSmart.out || '').trim()
            }
            if (!out) continue

            // record for learning
            speechSessionRawRef.current += (speechSessionRawRef.current ? '\n' : '') + segRaw
            speechSessionFormattedRef.current += (speechSessionFormattedRef.current ? '\n' : '') + out

            formattedPieces.push(out)
          }

          const appended = formattedPieces.join('\n').trim()
          if (!appended) return

          insertingFromSpeechRef.current = true
          setPrompt((prev) => {
            const base = prev || ''
            const sep = base && !base.endsWith('\n') ? '\n' : ''
            return base + sep + appended
          })
          setInputMode('voice')
        } catch (e) {
          console.error(e)
        }
      })()
    }

    r.onend = () => {
      // Chrome/Edge can end sessions automatically; keep alive until user presses Stop.
      if (shouldListenRef.current) {
        const wait = Math.min(2000, restartBackoffMsRef.current)
        restartBackoffMsRef.current = Math.min(2000, restartBackoffMsRef.current + 200)
        setTimeout(() => {
          try {
            r.start()
          } catch (_) {}
        }, wait)
      } else {
        setListening(false)
      }
    }

    r.onerror = (e) => {
      console.error(e)
      // Some errors are recoverable; keep alive if user didn't press stop.
      if (shouldListenRef.current) {
        setTimeout(() => {
          try {
            r.start()
          } catch (_) {}
        }, 500)
      } else {
        setListening(false)
      }
    }

    recognitionRef.current = r
  }, [setPrompt, setInputMode])

  // When the user presses Stop, we finalize a feedback anchor (prefix + dictation strings).
  useEffect(() => {
    if (listening) return
    if (!pendingFinalizeFeedbackRef.current) return
    pendingFinalizeFeedbackRef.current = false

    const raw = (speechSessionRawRef.current || '').trim()
    const formatted = (speechSessionFormattedRef.current || '').trim()
    if (!raw || !formatted) return

    const now = Date.now()

    // We assume dictation appended to the end of prompt (PromptPanel's behavior).
    const p = String(prompt || '')
    let prefix = null
    if (p.endsWith(formatted)) {
      prefix = p.slice(0, p.length - formatted.length)
    }

    speechFeedbackRef.current = {
      mode: modeRef.current === 'code' ? 'code' : 'qa',
      raw,
      formatted,
      prefix,
      createdAt: now
    }
  }, [listening, prompt])

  // If the user edits the prompt after dictation ended, submit feedback automatically.
  useEffect(() => {
    // Ignore prompt updates coming from speech insertion.
    if (insertingFromSpeechRef.current) {
      insertingFromSpeechRef.current = false
      return
    }

    const fb = speechFeedbackRef.current
    if (!fb || listening) return

    const ageMs = Date.now() - (fb.createdAt || 0)
    if (ageMs > 2 * 60 * 1000) {
      // too old; don't learn from random later edits
      speechFeedbackRef.current = null
      return
    }

    const p = String(prompt || '')
    if (fb.prefix == null || !p.startsWith(fb.prefix)) {
      return
    }

    const tail = p.slice(fb.prefix.length)
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim()
    if (!tail || norm(tail) === norm(fb.formatted)) return

    // Debounce to avoid sending feedback on every keystroke.
    if (feedbackTimerRef.current) {
      clearTimeout(feedbackTimerRef.current)
    }
    feedbackTimerRef.current = setTimeout(() => {
      submitSpeechFeedback({
        mode: fb.mode,
        raw_text: fb.raw,
        formatted_text: fb.formatted,
        final_text: tail
      })
        .then(() => {
          // Clear feedback after successful learn
          speechFeedbackRef.current = null
        })
        .catch(() => {
          // ignore
        })
    }, 900)
  }, [prompt, listening])

  function tryAutoSubmitAfterVoiceStop() {
    if (!autoSubmitVoiceRef.current) return
    if (disabled) return

    // Give React a moment to apply the last prompt update from speech.
    setTimeout(() => {
      try {
        const m = modeRef.current
        const p = String(promptRef.current || '').trim()
        if (!p) return

        // For code mode we require a project path. Q&A can be sent without it.
        if (m === 'code') {
          const proj = String(projectPathRef.current || '').trim()
          if (!proj) return
        }

        // Submit using the same action button handler.
        onStart?.()
      } catch (e) {
        console.error(e)
      }
    }, 150)
  }

  function toggleVoice() {
    const r = recognitionRef.current
    if (!r) {
      alert('Speech recognition is not supported in this browser.')
      return
    }

    if (listening) {
      // Stop: user explicitly ends dictation session.
      shouldListenRef.current = false
      pendingFinalizeFeedbackRef.current = true
      try {
        r.stop()
      } catch (_) {}
      setListening(false)

      // Optional: auto-submit current prompt when voice session ends.
      tryAutoSubmitAfterVoiceStop()
      return
    }

    // Start
    try {
      // Avoid TTS feedback loops if the app is currently speaking
      try {
        window.speechSynthesis?.cancel()
      } catch (_) {}

      shouldListenRef.current = true
      restartBackoffMsRef.current = 200
      listModeRef.current = { active: false, itemsLeft: 0, kind: 'bullet' }
      speechSessionRawRef.current = ''
      speechSessionFormattedRef.current = ''
      speechFeedbackRef.current = null
      pendingFinalizeFeedbackRef.current = false

      setListening(true)
      r.start()
    } catch (e) {
      console.error(e)
      setListening(false)
      alert('Failed to start speech recognition. Check mic permissions.')
    }
  }

  // ---------------------------------------------------------------------------
  // Action button state (fixes previous "startDisabled not defined" crash).
  // ---------------------------------------------------------------------------
  const isCode = mode === 'code'
  const trimmedPrompt = String(prompt || '').trim()
  const trimmedProject = String(projectPath || '').trim()

  const startDisabled =
    !!disabled ||
    (isCode ? !trimmedProject || !trimmedPrompt : !trimmedPrompt)

  const startLabel = disabled ? 'Running...' : isCode ? 'Start' : 'Send'

  return (
    <div className="card">
      <div className="grid">
        <div>
          <label>Mode</label>
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="code">Code pipeline</option>
            <option value="qa">Interactive Q&amp;A</option>
          </select>

          <div style={{ marginTop: 10 }}>
            <label>Model backend</label>
            <select
              value={llmBackend || 'local'}
              onChange={(e) => onLlmBackendChange?.(e.target.value)}
              disabled={disabled || modelSwitching}
            >
              <option value="local">Local</option>
              <option value="gemini">Gemini</option>
              <option value="openai">OpenAI</option>
            </select>
            <div className="small" style={{ marginTop: 6 }}>
              Active: <b>{llmBackend || 'local'}</b>
              {modelSwitching ? ' • switching...' : ''}
              {llmStatus?.models?.[llmBackend] ? ` • ${llmStatus.models[llmBackend]}` : ''}
            </div>
            {modelSwitchError ? (
              <div className="small" style={{ marginTop: 6 }}>
                <b>Model switch error:</b> {modelSwitchError}
              </div>
            ) : null}
          </div>

          {mode === 'qa' && (
            <>
              <div style={{ marginTop: 10 }}>
                <label>Q&amp;A orchestrator</label>
                <select value={qaOrchestrator} onChange={(e) => setQaOrchestrator(e.target.value)}>
                  {QA_ORCHESTRATORS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
                <div className="small" style={{ marginTop: 6 }}>
                  The HiveMasterOrchestrator routes your question to specialist orchestrators.
                </div>
              </div>
            </>
          )}

          <div style={{ marginTop: 10 }}>
            <label>Project path (optional for Q&amp;A)</label>
            <input
              value={projectPath}
              onChange={(e) => {
                setProjectPath(e.target.value)
                // project changes don't imply input mode change
              }}
              placeholder="e.g. hello or sample_projects/hello_py"
            />
            <div className="small" style={{ marginTop: 8 }}>
              Bare names resolve to <code>sample_projects/&lt;name&gt;</code>. Code mode creates missing sample_projects folders and edits them in place.
            </div>

            {mode === 'code' && (
              <div style={{ marginTop: 10 }}>
                <label style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={copyToWorkspace}
                    onChange={(e) => setCopyToWorkspace(e.target.checked)}
                  />
                  Copy non-sample project into backend workspace
                </label>
                <div className="small" style={{ marginTop: 6 }}>sample_projects are always edited directly, even when this is checked.</div>
              </div>
            )}

            <div style={{ marginTop: 10 }}>
              <label style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={autoSubmitVoice}
                  onChange={(e) => setAutoSubmitVoice(e.target.checked)}
                />
                Auto-submit when I press 🛑 Stop
              </label>
              <div className="small" style={{ marginTop: 6 }}>
                If enabled, voice dictation will be formatted, inserted into the prompt, and then automatically sent so the
                backend can respond (and Qwen can learn from the stored text).
              </div>
            </div>

            <div style={{ marginTop: 10 }}>
              <label style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                <input
                  type="checkbox"
                  checked={!!speakVoiceResponses}
                  onChange={(e) => setSpeakVoiceResponses?.(e.target.checked)}
                />
                Read answers aloud by default
              </label>
              <div className="small" style={{ marginTop: 6 }}>
                Leave this on to hear spoken replies for both typed and voice queries. Turn it off any time for text-only replies.
              </div>
            </div>
          </div>
        </div>

        <div>
          <label style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>{mode === 'code' ? 'NLP prompt' : 'Message'}</span>
            <button type="button" onClick={toggleVoice} disabled={disabled} title="Voice input">
              {listening ? '🛑 Stop' : '🎤 Speak'}
            </button>
          </label>

          <textarea
            value={prompt}
            onChange={(e) => {
              setPrompt(e.target.value)
              setInputMode('text')
            }}
            placeholder={mode === 'code' ? 'e.g. "Add a greet(name) function and tests for it."' : 'Speak or type your message…'}
          />

          <div className="small" style={{ marginTop: 6 }}>
            input_mode: <b>{inputMode}</b>
          </div>

          <div style={{ marginTop: 10 }}>
            <label>Speech recognition language</label>
            <select value={speechLang} onChange={(e) => setSpeechLang(e.target.value)} disabled={listening}>
              <option value="en-IN">English (India)</option>
              <option value="en-US">English (US)</option>
              <option value="en-GB">English (UK)</option>
              <option value="en-AU">English (Australia)</option>
            </select>
            <div className="small" style={{ marginTop: 6 }}>Uses multiple browser STT alternatives plus learned corrections.</div>
          </div>

          <div style={{ marginTop: 10 }}>
            <button onClick={() => onStart?.()} disabled={startDisabled}>
              {startLabel}
            </button>
          </div>

          {mode === 'qa' && (
            <div className="small" style={{ marginTop: 10 }}>
              Tip: If you want Qwen to learn from your voice conversations, keep using 🎤 Speak → 🛑 Stop (auto-submit),
              and correct any transcription mistakes in the textbox—those corrections will be learned for future dictation.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
