// bench/session-distribution.mjs — query the REAL sessions DB (~/.hermes/state.db,
// read-only) for the message_count distribution that frames the memory cells.
// Writes bench/session-distribution.json; render.mjs embeds it in the report.
//
// "message_count" here is DB messages (user/assistant/tool entries), the closest
// available unit to the bench's fixture rows. Not byte-identical semantics, but
// the same order of magnitude: both count transcript entries, not tokens.
//
// Usage: node session-distribution.mjs [path-to-state.db]
import { execFileSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

const here = dirname(fileURLToPath(import.meta.url))
const db = process.argv[2] ?? join(homedir(), '.hermes', 'state.db')

// No sqlite3 CLI on this box; the repo venv python is the canonical reader.
const py = join(here, '..', '.venv', 'bin', 'python')
const script = `
import sqlite3, json, math, sys
con = sqlite3.connect("file:" + sys.argv[1] + "?mode=ro", uri=True)
cur = con.cursor()
def pct(v, p):
    if not v: return None
    return v[max(0, min(len(v)-1, math.ceil(p/100*len(v))-1))]
def stats(rows):
    v = sorted(r[0] for r in rows if r[0] is not None)
    if not v: return None
    buckets = [(0,10),(10,25),(25,50),(50,100),(100,200),(200,300),(300,500),(500,1000),(1000,3000),(3000,None)]
    hist = []
    for lo,hi in buckets:
        c = sum(1 for x in v if x >= lo and (hi is None or x < hi))
        hist.append({"lo": lo, "hi": hi, "count": c})
    return {"n": len(v), "min": v[0], "max": v[-1], "mean": round(sum(v)/len(v),1),
            "p50": pct(v,50), "p75": pct(v,75), "p90": pct(v,90), "p95": pct(v,95), "p99": pct(v,99),
            "histogram": hist}
out = {
  "sources": dict(cur.execute("SELECT source, COUNT(*) FROM sessions GROUP BY source").fetchall()),
  "all": stats(cur.execute("SELECT message_count FROM sessions").fetchall()),
  "tui_cli": stats(cur.execute("SELECT message_count FROM sessions WHERE source IN ('tui','cli')").fetchall()),
  "tui_cli_top": [
    {"msgs": r[0], "tools": r[1] or 0, "source": r[2], "title": (r[3] or "")[:60]}
    for r in cur.execute("SELECT message_count, tool_call_count, source, title FROM sessions WHERE source IN ('tui','cli') ORDER BY message_count DESC LIMIT 10")
  ],
}
print(json.dumps(out, indent=2))
`
const json = execFileSync(py, ['-c', script, db], { encoding: 'utf8' })
const data = JSON.parse(json)
data.generated_at = new Date().toISOString()
data.db = db
const outFile = join(here, 'session-distribution.json')
writeFileSync(outFile, JSON.stringify(data, null, 2) + '\n')
const t = data.tui_cli
process.stdout.write(`tui+cli sessions n=${t.n}: p50=${t.p50} p75=${t.p75} p90=${t.p90} p95=${t.p95} p99=${t.p99} max=${t.max}\n→ ${outFile}\n`)
