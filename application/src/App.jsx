import { useState, useEffect, useRef } from 'react'
import './App.css'

const EMOTION_COLORS = {
  anger: '#e74c3c', annoyance: '#e67e22', disappointment: '#c0392b',
  disapproval: '#922b21', disgust: '#7d3c98', embarrassment: '#a93226',
  fear: '#6c3483', grief: '#5d6d7e', nervousness: '#884ea0',
  remorse: '#943126', sadness: '#2471a3',
  admiration: '#1e8bc3', amusement: '#27ae60', approval: '#2ecc71',
  caring: '#48c9b0', desire: '#f1948a', excitement: '#f39c12',
  gratitude: '#52be80', joy: '#f7dc6f', love: '#e91e63',
  optimism: '#3498db', pride: '#9b59b6', relief: '#1abc9c',
  curiosity: '#5dade2', confusion: '#aab7b8', neutral: '#7f8c8d',
  realization: '#85c1e9', surprise: '#f8c471',
}

function PoliticalBar({ lib, con, libPct, conPct }) {
  return (
    <div className="pol-bar-wrap">
      <div className="pol-bar">
        <div className="pol-segment liberal" style={{ width: `${libPct * 100}%` }}>
          {libPct > 0.08 && <span>{(libPct * 100).toFixed(1)}%</span>}
        </div>
        <div className="pol-segment conservative" style={{ width: `${conPct * 100}%` }}>
          {conPct > 0.08 && <span>{(conPct * 100).toFixed(1)}%</span>}
        </div>
      </div>
      <div className="pol-bar-labels">
        <span className="lib-label">Liberal ({lib})</span>
        <span className="con-label">Conservative ({con})</span>
      </div>
    </div>
  )
}

function EmotionChips({ top5 }) {
  const total = Object.values(top5).reduce((a, b) => a + b, 0)
  return (
    <div className="emotion-chips">
      {Object.entries(top5).map(([emotion, count]) => (
        <span
          key={emotion}
          className="chip"
          style={{ background: EMOTION_COLORS[emotion] || '#555' }}
        >
          {emotion} <em>{count}</em>
        </span>
      ))}
    </div>
  )
}

function BucketBar({ buckets }) {
  const bucketTotal = Object.values(buckets).reduce((a, b) => a + b, 0)
  const pct = (k) => ((buckets[k] || 0) / bucketTotal * 100).toFixed(1)
  return (
    <div className="bucket-bar-wrap">
      <div className="bucket-bar">
        <div className="bucket positive"  style={{ width: `${pct('positive')}%` }} />
        <div className="bucket neutral"   style={{ width: `${pct('neutral')}%` }} />
        <div className="bucket negative"  style={{ width: `${pct('negative')}%` }} />
      </div>
      <div className="bucket-labels">
        <span className="pos">Positive {pct('positive')}%</span>
        <span className="neu">Neutral {pct('neutral')}%</span>
        <span className="neg">Negative {pct('negative')}%</span>
      </div>
    </div>
  )
}

function ResultsPanel({ result }) {
  const { topic_names, posts_found, total_comments, political, emotions, summaries,
          dominant_pol, dominant_pct, avg_confidence } = result

  return (
    <div className="results">
      <div className="results-header">
        <h2>{topic_names}</h2>
        <div className="meta">
          {posts_found} posts · {total_comments} comments ·{' '}
          dominant: <strong className={dominant_pol === 'Liberal' ? 'lib' : 'con'}>
            {dominant_pol} ({(dominant_pct * 100).toFixed(1)}%)
          </strong>
        </div>
      </div>

      <div className="card">
        <h3>Political Leaning</h3>
        <PoliticalBar
          lib={political.Liberal} con={political.Conservative}
          libPct={political.lib_pct} conPct={political.con_pct}
        />
      </div>

      <div className="summaries">
        {['Liberal', 'Conservative'].map((pol) => (
          <div key={pol} className={`summary-card ${pol === 'Liberal' ? 'lib-card' : 'con-card'}`}>
            <div className="summary-text">"{summaries[pol]}"</div>
          </div>
        ))}
      </div>

      <div className="group-cards">
        {['Liberal', 'Conservative'].map((pol) => (
          <div key={pol} className={`group-card ${pol === 'Liberal' ? 'lib-border' : 'con-border'}`}>
            <h3>{pol}</h3>
            <BucketBar buckets={emotions[pol].buckets} />
            <h4>Top Emotions</h4>
            <EmotionChips top5={emotions[pol].top5} />
          </div>
        ))}
      </div>

      <div className="confidence-row">
        Avg model confidence: <strong>{(avg_confidence * 100).toFixed(1)}%</strong>
      </div>
    </div>
  )
}

export default function App() {
  const [topics, setTopics]       = useState([])
  const [selectedId, setSelectedId] = useState('')
  const [log, setLog]             = useState([])
  const [result, setResult]       = useState(null)
  const [running, setRunning]     = useState(false)
  const esRef = useRef(null)
  const logEndRef = useRef(null)

  useEffect(() => {
    fetch('/api/topics')
      .then(r => r.json())
      .then(setTopics)
  }, [])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [log])

  function startAnalysis() {
    if (!selectedId) return
    if (esRef.current) esRef.current.close()

    setLog([])
    setResult(null)
    setRunning(true)

    const es = new EventSource(`/api/analyze?topic_id=${selectedId}`)
    esRef.current = es

    es.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'status') {
        setLog(prev => [...prev, msg.message])
      } else if (msg.type === 'match') {
        setLog(prev => [...prev, `Match ${msg.count}/${msg.target}: ${msg.title.slice(0, 60)}`])
      } else if (msg.type === 'result') {
        setResult(msg.data)
        setRunning(false)
        es.close()
      } else if (msg.type === 'error') {
        setLog(prev => [...prev, `Error: ${msg.message}`])
        setRunning(false)
        es.close()
      }
    }

    es.onerror = () => {
      setLog(prev => [...prev, 'Connection error.'])
      setRunning(false)
      es.close()
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Reddit Political Sentiment Analyzer</h1>
        <p>Scrapes r/politics and analyzes political + emotional tone by topic</p>
      </header>

      <div className="controls card">
        <label htmlFor="topic-select">Select a topic</label>
        <select
          id="topic-select"
          value={selectedId}
          onChange={e => setSelectedId(e.target.value)}
          disabled={running}
        >
          <option value="">— choose a topic —</option>
          {topics.map(t => (
            <option key={t.id} value={t.id}>{t.id}: {t.name}</option>
          ))}
        </select>
        <button onClick={startAnalysis} disabled={!selectedId || running}>
          {running ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>

      {(log.length > 0) && (
        <div className="log card">
          <h3>Progress</h3>
          <div className="log-body">
            {log.map((line, i) => <div key={i} className="log-line">{line}</div>)}
            {running && <div className="log-line blink">▋</div>}
            <div ref={logEndRef} />
          </div>
        </div>
      )}

      {result && <ResultsPanel result={result} />}
    </div>
  )
}
