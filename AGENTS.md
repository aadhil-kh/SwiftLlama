# Building Agents with SwiftLlama

SwiftLlama is a lightweight, Swift-friendly wrapper around llama.cpp for local LLM inference. This guide distills the codebase into actionable patterns for building “agents”: stateful assistants, streaming chat UIs, and tool-augmented loops.

- Core types: `SwiftLlama`, `Prompt`, `Chat`, `Configuration`, `StopToken`
- Concurrency: `@SwiftLlamaActor` ensures serialized access to the model
- Platforms: macOS, iOS (see TestProjects for sample apps)


## Architecture at a glance

- llama.cpp C bindings (`import llama`) expose low-level APIs
- `LlamaModel`: owns model/context, tokenization, sampling, decode loop
- `SwiftLlama`: high-level façade for prompts, streaming, stop-token handling, and optional session memory
- `Prompt`: builds model-specific prompt strings (llama, llama3, alpaca, chatML, mistral, phi, gemma)
- `Chat`: represents (user, bot) turns used to build prompt history
- `Session`: rolling chat memory when `sessionSupport` is enabled
- `Configuration`: runtime parameters (context length, temperature, stop tokens, etc.)
- `StopToken`: suggested stop sequences per model family


## The agent contract

- Input
  - Model path (GGUF file)
  - `Prompt(type:, systemPrompt:, userMessage:, history:)`
  - `Configuration` (e.g., `nCTX`, `temperature`, `stopTokens`, `maxTokenCount`)
- Output
  - Streaming tokens (AsyncStream or Combine) or a single `String`
- Errors
  - `SwiftLlamaError.decodeError`, `SwiftLlamaError.others(String)`
- Completion
  - Stops on EOS, `maxTokenCount`, or a matched stop token; library strips stop tokens from output


## Prompting correctly

Pick the `Prompt.Type` that matches your model family:

- `.llama3` uses `<|start_header_id|>…<|end_header_id|>` and `<|eot_id|>`
- `.chatML` uses `<|im_start|>`/`<|im_end|>`
- `.phi` uses `<|user|>`, `<|assistant|>`, `<|end|>`
- `llama`, `alpaca`, `mistral`, `gemma` each have their own template implementations

History: only the most recent `Configuration.historySize` turns (default 5) are included. Provide `history` explicitly or enable `sessionSupport` to let the library manage it.


## Stop tokens 101

Set `Configuration.stopTokens` to cleanly terminate generation and avoid leaking control tokens:

- Llama 3 → `StopToken.llama3` (e.g., `<|eot_id|>`) 
- Llama classic → `StopToken.llama` (e.g., `[/INST]`)
- ChatML → `StopToken.chatML` (e.g., `<|im_end|>`) 
- Phi → `StopToken.phi` (e.g., `<|end|>`) 

The stream logic buffers a short tail so split stop sequences are detected across token boundaries, then trims the stop tokens from output.


## Concurrency and sessions

- Public APIs on `SwiftLlama` are annotated with `@SwiftLlamaActor`; calls are serialized to a single global actor to protect the underlying context
- Treat one `SwiftLlama` instance as single-consumer; avoid concurrent `start` calls on the same instance
- `sessionSupport: true` enables rolling history: after completion, the pair `(user:lastPrompt.userMessage, bot:response)` is appended to memory


## Using the APIs

1) AsyncStream (recommended)

```
let llama = try SwiftLlama(modelPath: modelPath, modelConfiguration: config)
let prompt = Prompt(type: .llama3, systemPrompt: "You are helpful.", userMessage: "Hello!")

var text = ""
for try await delta in await llama.start(for: prompt, sessionSupport: true) {
    text += delta
}
```

2) Combine Publisher

```
let pub = await llama.start(for: prompt, sessionSupport: true)
var cancel = Set<AnyCancellable>()
pub.sink(receiveCompletion: { _ in }, receiveValue: { delta in
    print(delta, terminator: "")
}).store(in: &cancel)
```

3) Non-streaming

```
let full: String = try await llama.start(for: prompt, sessionSupport: true)
```


## Configuration quick reference

- `nCTX`: context tokens (must not exceed model’s train context)
- `temperature`: creativity (lower for precision)
- `maxTokenCount`: hard cap on generated tokens
- `batchSize`: llama.cpp batch size for decode
- `stopTokens`: sequences that terminate generation
- Threads: derived from CPU count; GPU layers forced to 0 on simulator


## Build a stateful chat agent

```
let config = Configuration(nCTX: 4096, temperature: 0.7, stopTokens: StopToken.llama3)
let llama = try SwiftLlama(modelPath: modelPath, modelConfiguration: config)

func ask(_ user: String) async throws -> String {
    let p = Prompt(type: .llama3, systemPrompt: "You are a helpful assistant.", userMessage: user)
    return try await llama.start(for: p, sessionSupport: true)
}

let a = try await ask("Who are you?")
let b = try await ask("Summarize the previous answer.")
```


## Tool-augmented loop (ReAct-style sketch)

```
struct ToolResult { let name: String; let output: String }

func step(history: inout [Chat], user: String) async throws -> (String, ToolResult?) {
    let p = Prompt(type: .llama3,
                   systemPrompt: "Use TOOLS when needed. Format: Action: <tool> <args> or Final: <answer>",
                   userMessage: user,
                   history: history)

    let answer = try await llama.start(for: p, sessionSupport: false)
    if let (tool, args) = parseAction(answer) {
        let obs = await runTool(name: tool, args: args)
        history.append(Chat(user: user, bot: "Observation: \(obs)"))
        return (answer, ToolResult(name: tool, output: obs))
    } else {
        history.append(Chat(user: user, bot: answer))
        return (answer, nil)
    }
}
```

Note: choose stop tokens and output formats so the model won’t truncate tool directives mid-token.


## SwiftUI streaming sketch

```
@Observable
final class ChatViewModel {
    private let llama: SwiftLlama
    var text = ""

    init(llama: SwiftLlama) { self.llama = llama }

    func send(_ message: String) {
        let p = Prompt(type: .llama3, systemPrompt: "You are helpful.", userMessage: message)
        Task { [weak self] in
            guard let self else { return }
            for try await delta in await llama.start(for: p, sessionSupport: true) {
                self.text += delta
            }
        }
    }
}
```


## Troubleshooting

- Prints control tokens → add matching `stopTokens` for your prompt template
- Stops too early → remove over-eager stop tokens; ensure they don’t appear in normal text
- Garbled characters → stream is UTF-8; make sure your UI component appends as UTF-8 text
- Session not remembered → ensure `sessionSupport: true` and that you consume the stream to completion


## Limitations and ideas

- No explicit cancellation API in `SwiftLlama` (today: stop consuming the stream). Consider adding cooperative cancel if you need it.
- `Configuration.historySize` is static; refactor to make it configurable if required.
- Function calling/tooling: build on top with your own schemas/JSON parsing helpers.


## See also

- `README.md` for install/usage basics
- `TestProjects/` for working iOS/macOS/CLI samples
- `Sources/SwiftLlama/Models/Prompt.swift` for exact prompt encodings
