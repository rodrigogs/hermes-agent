# Client tool catalog + tool-callback channel for the API server

**Status: IMPLEMENTADO neste card.** Código em
`gateway/platforms/api_server_client_tools.py` + plumagem em
`gateway/platforms/api_server.py` (intercept em
`agent/agent_runtime_helpers.py::invoke_tool` e
`model_tools.py::handle_function_call`); contrato em
`tests/gateway/test_api_server_client_tools.py` (24 testes). Destrava o relaying de tools do
Trama (card `t_4b069539`, aberto a partir do board trama `t_c912b8f9`; design
do lado Trama em `docs/superpowers/specs/2026-08-31-relaying-de-tools.md`,
decisões D1–D4).

## 1. O problema

Um backend que roda o próprio loop (hoje: o Trama) precisa chamar uma tool que
executa no HOST do cliente. O API server do Hermes rodava o loop inteiro com
o toolset configurado no servidor (`platform_toolsets.api_server`): não havia
campo `tools` no corpo de `/api/sessions/{id}/chat[/stream]` e não havia canal
para o cliente registrar callbacks — o registro normativo do Trama
(`2026-08-09-tool-use-entre-agentes.md`) media isso como bloqueio.

## 2. Decisões

**D1 — Catálogo por requisição, não registro de servidor.** O cliente manda
`tools: [{name, description, parameters}]` (formato OpenAI) no corpo do chat.
O catálogo vale só para aquela requisição; nada toca o registry global de
tools nem o `platform_toolsets` de outros canais. Risco de colisão é tratado
por prefixo de namespace (`trama_` no uso real): nomes que colidem com uma
tool nativa são REJEITADOS na validação (400), nunca sobrescrevem.

**D2 — O molde é o approval channel do `/v1/runs`** (`gateway/platforms/api_server_runs.py`
+ `tools/approval.py`): a thread do turno estaciona em `threading.Event`
enquanto o cliente não responde; o cliente resolve via HTTP POST; o unregister
no `finally` acorda todas as threads pendentes (fail-closed, sem hang). A
`_ApprovalEntry` provou o padrão em produção — é reuso, não invenção.

**D3 — A bridge entra no `handle_function_call`, não na borda HTTP.** A tool
relayada é despachada pelo MESMO funil das tools nativas. Em um turno real
(qualquer executor) o funil do executor — request middleware, pre_tool_call
hooks, guardrails, post_tool_call terminal — vê a tool relayada normalmente;
o intercept da bridge acontece DENTRO do dispatcher
(`handle_function_call`/`invoke_tool`), antes do lookup no registry. Nuance:
quando o dispatcher é chamado FORA de um turno (chamada direta, sem
executor), o intercept é a primeira checagem — post_tool_call dispara, mas
request/execution middleware e pre_tool_call hooks não rodam (o executor é
quem os executa; uma chamada direta não tem executor). Não há segunda classe
de tools invisível a hooks/audit no caminho de turno.

**D4 — Injection pattern = context engine** (`agent/agent_init.py:2959-2994`):
schemas anexados a `agent.tools` + `agent.valid_tool_names` APÓS o snapshot
do registry, com dedup de nomes. A bridge é um objeto da própria instância do
agente (não estado de processo), então agentes sem catálogo são intocados.

**D5 — Fail-closed com deadline.** Chamada de tool sem canal registrado →
`tool_error` estruturado imediatamente (o modelo se corrige no próprio loop).
Canal registrado mas sem resposta → deadline (default 120 s, configurável
`api_server.client_tools_timeout_seconds`) → `tool_timeout` para o modelo.
Desconexão do turno → unregister acorda as pendentes com erro.

**D6 — Lifecycle acoplado à sessão HTTP, registrada por request.** O canal é
registrado em `_run_agent()` (ponto único que cobre `/chat` e `/chat/stream`)
com chave = `id` da bridge (uma por requisição) e desregistrado no `finally`
que já existe lá. Sem estado que sobreviva à requisição.

**D7 — Segurança.** O catálogo não é código: só schema declarativo (o handler
é a ponte HTTP, nunca algo do cliente). Schemas validados (nome, tamanho,
JSON Schema object); `parameters` não-objeto rejeitado. Obedece AGENTS.md
Footprint Ladder: zero novas model tools; o custo é pago só pela requisição
que declara o catálogo.

## 3. Contrato HTTP (adições)

**POST /api/sessions/{id}/chat e /chat/stream** — novo campo opcional:

```json
{
  "message": "...",
  "tools": [
    {"type": "function",
     "function": {"name": "trama_navegar",
                  "description": "Navega o browser do host",
                  "parameters": {"type": "object",
                                 "properties": {"url": {"type": "string"}},
                                 "required": ["url"]}}}
  ]
}
```

- Também aceita a forma curta `[{name, description, parameters}]`.
- Quando `tools` está presente, o turno entra em **split runtime**:
  `client_tools` vira o único catálogo acrescentado (as toolsets nativas do
  servidor continuam disponíveis conforme config — o cliente que não quiser
  exposição não usa o canal; escopo fino fica para um card futuro).
- Quando `tools` está ausente ou vazio: comportamento idêntico ao atual
  (regressão zero para clientes existentes).

**Novos eventos SSE** (apenas em `/chat/stream` com catálogo):

- `client_tool.call` — `{tool_call_id, name, arguments}` — a chamada que o
  cliente deve executar no host.
- O resultado volta SEMPRE por **POST /api/sessions/{id}/chat/tool-result**
  `{tool_call_id, output, is_error}` — no modo síncrono e no stream. O turno
  fica estacionado (até `client_tools_timeout_seconds`, default 120 s)
  enquanto o resultado não chega; não há campo `tool_results` no corpo do
  `/chat`. Idempotente: resposta a `tool_call_id` desconhecido/antigo → 409
  estável (`tool_call_not_pending`), nunca corrói o turno.
- Um segundo `/chat` com `tools` na MESMA sessão enquanto o canal do turno
  anterior está vivo → 409 estável `client_tools_channel_active` (o handler
  síncrono responde HTTP 409; o stream emite evento `error` com o mesmo
  `code`). Registrar por cima orfaria as chamadas pendentes do turno
  anterior até o timeout.

**Response do /chat síncrono** — forma normal do turno OpenAI: o modelo
executa a(s) tool(s) dentro do MESMO turno (a thread do agente estaciona na
bridge até o resultado chegar pelo POST acima, ou o timeout estourar) e a
resposta carrega o `final_response` já incorporando os outputs relayados.
Não há término antecipado com `tool_calls` pendentes no corpo.

**`GET /v1/capabilities`** — `features.session_client_tools` (bool) +
`endpoints.session_tool_result`.

## 4. Fluxo (stream)

1. Request com `tools` → handler valida catálogo (D1/D7).
2. `_run_agent(..., client_tools=catalog)` → `_create_agent` injeta schemas
   (D4) e instala a bridge na instância (D3).
3. Modelo chama `trama_navegar` → `_execute_tool_calls` → `invoke_tool` →
   `handle_function_call` → bridge reconhece o nome → serializa
   `client_tool.call` no stream/cola → thread estaciona em `Event` (D2).
4. Cliente executa no host e POSTa o resultado → handler resolve a entrada →
   thread acorda → resultado volta ao loop como tool result normal.
5. Fim do turno: `unregister` no `finally` (D2/D5/D6).

No modo síncrono o contrato é o MESMO estacionamento: a conexão `/chat`
permanece aberta enquanto a bridge aguarda o POST do resultado (bounded pelo
timeout de 120 s; o endpoint responde assim que o turno termina). O
estacionamento é por THREAD do agente, dentro do turno — nunca um fim
antecipado do turno com `tool_calls` pendentes no corpo da resposta.

## 5. Arquivos

| Arquivo | Mudança |
|---|---|
| `gateway/platforms/api_server_client_tools.py` | NOVO — módulo da bridge: validação de catálogo, `ClientToolBridge` (Event-based), handlers das rotas, resolução |
| `gateway/platforms/api_server.py` | campo `tools` em ambos handlers de chat; plumagem `client_tools` em `_run_agent`/`_create_agent`; capabilities; rota `tool-result`; 409 `client_tools_channel_active` em `/chat` (HTTP) e `/chat/stream` (evento `error`) |
| `agent/agent_runtime_helpers.py` | `invoke_tool` consulta a bridge do agente antes do registry |
| `model_tools.py` | `handle_function_call` idem (mesma checagem, ponto único) |
| `agent/tool_executor.py` | executor sequencial passa `agent=` ao `handle_function_call` — o intercept da bridge é agent-gated; sem isso, o turno com UMA tool call (o caso típico do relaying) nunca estaciona na bridge |
| `tests/gateway/test_api_server_client_tools.py` | NOVO — suite do contrato |

`agent/conversation_loop` NÃO muda: a bridge intercepta abaixo dele, no
dispatcher, e o funil do executor é agnóstico à origem da tool.

## 6. Testes (contrato, não snapshot)

1. Request sem `tools` → turno idêntico ao atual (regressão).
2. Catálogo inválido (nome duplicado, colisão com tool nativa, schema ruim) →
   400 estável.
3. Modelo chama a tool do cliente: stream emite `client_tool.call`; resultado
   POSTado retorna ao loop; resposta final cita o output.
4. Timeout sem resposta → turno continua com `tool_timeout` (modelo se corrige).
5. Sem canal (modo síncrono pós-turno) → erro estruturado imediato.
6. tool-result desconhecido → 409 estável, turno intacto.
7. Two-turn síncrono: assistant tool_calls persistidas; segundo /chat com
   `tool_results` fecha o par; alternation preservada.
8. Regressão de alternation: assistant-with-tool_calls nunca fica sem par de
   resultados no histórico (regra #48879/#52592 mantida).

Execução: `scripts/run_tests.sh tests/gateway/test_api_server_client_tools.py`.
