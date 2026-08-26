unit Prism.RestServer;

{ REST interface, compatible with OpenAI and Ollama clients.

  OpenAI-compatible:
    GET  /v1/models
    POST /v1/chat/completions   (stream: SSE "data: ...")
    POST /v1/completions
  Ollama-compatible:
    GET  /api/tags, /api/version
    POST /api/generate, /api/chat  (stream: NDJSON)
  Prism extensions:
    GET  /health                 Status + model info
    POST /api/train              Feed a training sample (text/multimodal)

  Timing in every answer:
    Ollama endpoints report total_duration / load_duration /
    prompt_eval_duration / eval_duration in nanoseconds, as Ollama does.
    OpenAI endpoints carry "x_timings" (total_ms, prompt_ms, eval_ms,
    tokens_per_second). While streaming, both arrive in the final packet.

  Additional fields in chat requests:
    "verify": true  -> self-verification, result in "x_verification"

  Implemented with TIdHTTPServer (Indy, part of Delphi). }

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs, System.JSON,
  System.DateUtils, System.NetEncoding, System.IOUtils,
  System.Generics.Collections,
  IdHTTPServer, IdCustomHTTPServer, IdContext, IdGlobal,
  Prism.Types, Prism.Model, Prism.Streaming, Prism.Tokenizer,
  Prism.Inference, Prism.Llama, Prism.Verify, Prism.Multimodal,
  Prism.Train, Prism.Gpu, Prism.Laws;

type
  TServerOptions = record
    Port: Integer;
    ModelPath: string;       // .prism or .gguf
    TokenizerPath: string;   // only for .prism
    StreamLayers: Integer;   // 0 = load the whole model into RAM
    MaxCachedExperts: Integer;
    CtxOverride: Integer;    // cap the GGUF context window (RAM)
    VerifyDefault: Boolean;
    TrainEnabled: Boolean;
    CorpusPath: string;
    Template: string;        // auto|prism|chatml|llama2|plain
    UseGpu: Boolean;
    GpuBudgetMB: Integer;
    class function Default: TServerOptions; static;
  end;

  TPrismRestServer = class
  private
    FOpts: TServerOptions;
    FHttp: TIdHTTPServer;
    FBackend: TLlmBackend;
    FVerifier: TVerifier;
    FGenLock: TCriticalSection;
    { One generator for the whole server, not one per request. Creating one
      builds a fresh engine, and an engine allocates the full KV cache --
      2 x NLayers x CtxLen x KvDim floats, which is 2.4 GB for a 3B model at
      the default 32768 context. Doing that per HTTP request cost ~0.3 s of
      allocation alone. Safe to share because every generation runs inside
      FGenLock; see SharedGenerator. }
    FGen: TGenerator;
    FTrainSvc: TTrainingService;
    FLog: TProc<string>;
    FTemplate: TChatTemplate;
    function SharedGenerator: TGenerator;
    procedure Log(const S: string);
    procedure DoCommand(AContext: TIdContext;
      ARequestInfo: TIdHTTPRequestInfo; AResponseInfo: TIdHTTPResponseInfo);
    procedure DoCommandOther(AContext: TIdContext;
      ARequestInfo: TIdHTTPRequestInfo; AResponseInfo: TIdHTTPResponseInfo);
    function ReadBody(ARequestInfo: TIdHTTPRequestInfo): string;
    procedure SendJson(AResponseInfo: TIdHTTPResponseInfo; Obj: TJSONObject;
      Code: Integer = 200);
    procedure SendError(AResponseInfo: TIdHTTPResponseInfo; Code: Integer;
      const Msg: string);
    procedure BeginStream(AResponseInfo: TIdHTTPResponseInfo;
      const ContentType: string);
    procedure StreamChunk(AContext: TIdContext; const S: string);
    procedure StreamEnd(AContext: TIdContext);
    procedure ParseMessages(Body: TJSONObject; out Msgs: TChatMessages;
      out LastUser: string);
    function ParseSampling(Body: TJSONObject): TSamplingParams;
    procedure HandleChat(AContext: TIdContext; Body: TJSONObject;
      AResponseInfo: TIdHTTPResponseInfo; OllamaStyle: Boolean);
    procedure HandleCompletions(AContext: TIdContext; Body: TJSONObject;
      AResponseInfo: TIdHTTPResponseInfo);
    procedure HandleGenerate(AContext: TIdContext; Body: TJSONObject;
      AResponseInfo: TIdHTTPResponseInfo);
    procedure HandleTrain(Body: TJSONObject;
      AResponseInfo: TIdHTTPResponseInfo);
    procedure HandleModels(AResponseInfo: TIdHTTPResponseInfo);
    procedure HandleTags(AResponseInfo: TIdHTTPResponseInfo);
    procedure HandleHealth(AResponseInfo: TIdHTTPResponseInfo);
    procedure HandleLawsList(AResponseInfo: TIdHTTPResponseInfo;
      const Query: string);
    procedure HandleCalc(Body: TJSONObject;
      AResponseInfo: TIdHTTPResponseInfo);
    procedure HandleLawEval(Body: TJSONObject;
      AResponseInfo: TIdHTTPResponseInfo);
  public
    constructor Create(const AOpts: TServerOptions; const ALog: TProc<string>);
    destructor Destroy; override;
    procedure Start;
    procedure Stop;
    property Backend: TLlmBackend read FBackend;
  end;

implementation

function NewId(const Prefix: string): string;
var
  G: TGUID;
begin
  CreateGUID(G);
  Result := Prefix + Copy(GUIDToString(G).Replace('{', '').Replace('}', '')
    .Replace('-', '').ToLower, 1, 24);
end;

function UnixNow: Int64;
begin
  Result := DateTimeToUnix(Now, False);
end;

function IsoNowUtc: string;
begin
  Result := FormatDateTime('yyyy-mm-dd"T"hh:nn:ss"Z"',
    UnixToDateTime(UnixNow, True));
end;

function GetStr(Obj: TJSONObject; const Key, Def: string): string;
var
  V: TJSONValue;
begin
  Result := Def;
  if Obj = nil then
    Exit;
  V := Obj.GetValue(Key);
  if V is TJSONString then
    Result := TJSONString(V).Value;
end;

function GetNum(Obj: TJSONObject; const Key: string; Def: Double): Double;
var
  V: TJSONValue;
begin
  Result := Def;
  if Obj = nil then
    Exit;
  V := Obj.GetValue(Key);
  if V is TJSONNumber then
    Result := TJSONNumber(V).AsDouble;
end;

function GetBool(Obj: TJSONObject; const Key: string; Def: Boolean): Boolean;
var
  V: TJSONValue;
begin
  Result := Def;
  if Obj = nil then
    Exit;
  V := Obj.GetValue(Key);
  if V is TJSONBool then
    Result := TJSONBool(V).AsBoolean;
end;

{ Haengt die Zeitmessung an eine Antwort.

  Ollama hat dafuer bereits Felder, und zwar in NANOSEKUNDEN: total_duration,
  load_duration, prompt_eval_duration, eval_duration. Clients rechnen ihre
  Token/s daraus aus, also werden genau diese Namen bedient statt eigener --
  sonst zeigt ein Ollama-Client neben Prism eine leere Geschwindigkeit an.
  load_duration ist 0: das Modell steht beim ersten Request bereits.

  Fuer OpenAI gibt es kein solches Feld. Dort gilt die Prism-Konvention der
  x_-Erweiterungen (wie x_verification, x_areas), in Millisekunden, weil das
  die Einheit ist, in der ueber HTTP-Antwortzeiten geredet wird. }
procedure AddTimings(Root: TJSONObject; const Usage: TUsage;
  OllamaStyle: Boolean);
var
  T: TJSONObject;
begin
  if OllamaStyle then
  begin
    Root.AddPair('total_duration',
      TJSONNumber.Create(Round(Usage.TotalMs * 1e6)));
    Root.AddPair('load_duration', TJSONNumber.Create(Int64(0)));
    Root.AddPair('prompt_eval_duration',
      TJSONNumber.Create(Round(Usage.PrefillMs * 1e6)));
    Root.AddPair('eval_duration',
      TJSONNumber.Create(Round(Usage.DecodeMs * 1e6)));
  end
  else
  begin
    T := TJSONObject.Create;
    T.AddPair('total_ms', TJSONNumber.Create(Usage.TotalMs));
    T.AddPair('prompt_ms', TJSONNumber.Create(Usage.PrefillMs));
    T.AddPair('eval_ms', TJSONNumber.Create(Usage.DecodeMs));
    T.AddPair('tokens_per_second',
      TJSONNumber.Create(Usage.TokensPerSecond));
    Root.AddPair('x_timings', T);
  end;
end;

function TemplateFromString(const S: string): TChatTemplate;
begin
  if SameText(S, 'prism') then
    Result := ctPrism
  else if SameText(S, 'chatml') then
    Result := ctChatML
  else if SameText(S, 'llama2') then
    Result := ctLlama2
  else if SameText(S, 'plain') then
    Result := ctPlain
  else
    Result := ctAuto;
end;

{ TServerOptions }

class function TServerOptions.Default: TServerOptions;
begin
  Result.Port := 11434;
  Result.ModelPath := '';
  Result.TokenizerPath := '';
  Result.StreamLayers := 0;
  Result.MaxCachedExperts := 8;
  Result.CtxOverride := 0;
  Result.VerifyDefault := False;
  Result.TrainEnabled := False;
  Result.CorpusPath := 'corpus.bin';
  Result.Template := 'auto';
  Result.UseGpu := False;
  Result.GpuBudgetMB := 0;
end;

{ TPrismRestServer }

constructor TPrismRestServer.Create(const AOpts: TServerOptions;
  const ALog: TProc<string>);
var
  Ext, GpuInfo: string;
  Prov: TWeightsProvider;
  Tok: TTokenizer;
  Full: TFullWeights;
begin
  inherited Create;
  FOpts := AOpts;
  FLog := ALog;
  FGenLock := TCriticalSection.Create;
  FGen := nil;
  FTemplate := TemplateFromString(AOpts.Template);

  if AOpts.UseGpu then
  begin
    { Bring the GPU up BEFORE the model is loaded: weights are uploaded lazily
      on first use, so the backend has to exist by the time the first token
      runs through the layers. }
    if TryInitGpuBackend(GpuInfo, AOpts.GpuBudgetMB,
         procedure(S: string) begin Log(S); end) then
    begin
      Log('GPU backend active: ' + GpuInfo);
      if not GpuQuantizedActive then
        Log('Note: quantized (GGUF) kernels are NOT on the GPU with this ' +
            'backend - they stay on the CPU.');
    end
    else
      Log('GPU not available (' + GpuInfo + '), using CPU.');
  end;

  Ext := LowerCase(ExtractFileExt(AOpts.ModelPath));
  if Ext = '.gguf' then
  begin
    Log('Loading GGUF model: ' + AOpts.ModelPath);
    FBackend := TLlamaBackend.Create(AOpts.ModelPath, AOpts.CtxOverride,
      AOpts.StreamLayers, procedure(S: string) begin Log(S); end);
    if FTemplate <> ctAuto then
      TLlamaBackend(FBackend).Template := FTemplate;
    Log(Format('GGUF ready: %s (arch %s, vocab %d, context %d)',
      [FBackend.ModelName, TLlamaBackend(FBackend).Model.Gguf.Arch,
       TLlamaBackend(FBackend).Model.Cfg.Vocab,
       TLlamaBackend(FBackend).Model.Cfg.CtxLen]));
  end
  else
  begin
    if not FileExists(AOpts.TokenizerPath) then
      raise Exception.Create('Tokenizer file missing: ' + AOpts.TokenizerPath);
    Tok := TTokenizer.Create;
    Tok.LoadFromFile(AOpts.TokenizerPath);
    if AOpts.StreamLayers > 0 then
    begin
      Log(Format('Loading Prism model (streaming, %d layers in cache): %s',
        [AOpts.StreamLayers, AOpts.ModelPath]));
      Prov := TLayerStore.Create(AOpts.ModelPath, AOpts.StreamLayers,
        AOpts.MaxCachedExperts);
    end
    else
    begin
      Log('Loading Prism model (fully in RAM): ' + AOpts.ModelPath);
      Full := TFullWeights.Create;
      Full.LoadFromFile(AOpts.ModelPath);
      Prov := Full;
    end;
    Log('Configuration: ' + Prov.Config.ToString);
    FBackend := TPrismBackend.Create(Prov, Tok,
      TPath.GetFileNameWithoutExtension(AOpts.ModelPath));
    if AOpts.TrainEnabled then
    begin
      if Prov is TFullWeights then
      begin
        FTrainSvc := TTrainingService.Create(TFullWeights(Prov), Tok,
          FGenLock, AOpts.ModelPath, nil,
          procedure(S: string) begin Log(S); end);
        Log('Online training active (POST /api/train).');
      end
      else
        Log('Note: online training requires --stream-layers 0; disabled.');
    end;
  end;

  FVerifier := TVerifier.Create(FBackend);

  FHttp := TIdHTTPServer.Create(nil);
  FHttp.DefaultPort := AOpts.Port;
  FHttp.OnCommandGet := DoCommand;
  FHttp.OnCommandOther := DoCommandOther;
  { Always pass the body through as PostStream - otherwise Indy consumes
    form-encoded POSTs (clients without Content-Type: application/json) }
  FHttp.ParseParams := False;
  FHttp.ServerSoftware := 'Prism/' + PRISM_VERSION;
end;

destructor TPrismRestServer.Destroy;
begin
  Stop;
  if FTrainSvc <> nil then
  begin
    FTrainSvc.Shutdown;
    FTrainSvc.Free;
  end;
  FHttp.Free;
  FVerifier.Free;
  FBackend.Free;
  FGen.Free;
  FGenLock.Free;
  inherited;
end;

procedure TPrismRestServer.Log(const S: string);
begin
  if Assigned(FLog) then
    FLog(S);
end;

procedure TPrismRestServer.Start;
begin
  FHttp.Active := True;
  Log(Format('Prism server running on port %d', [FOpts.Port]));
  Log('  OpenAI-API:  POST http://localhost:' + IntToStr(FOpts.Port) +
    '/v1/chat/completions');
  Log('  Ollama-API:  POST http://localhost:' + IntToStr(FOpts.Port) +
    '/api/chat');
end;

procedure TPrismRestServer.Stop;
begin
  if (FHttp <> nil) and FHttp.Active then
    FHttp.Active := False;
end;

function TPrismRestServer.ReadBody(ARequestInfo: TIdHTTPRequestInfo): string;
var
  SS: TStringStream;
begin
  Result := '';
  if ARequestInfo.PostStream <> nil then
  begin
    SS := TStringStream.Create('', TEncoding.UTF8);
    try
      ARequestInfo.PostStream.Position := 0;
      SS.CopyFrom(ARequestInfo.PostStream, 0);
      Result := SS.DataString;
    finally
      SS.Free;
    end;
  end;
  { Clients without "Content-Type: application/json": for form-urlencoded,
    Indy places the body in FormParams instead of PostStream }
  if (Result = '') and (ARequestInfo.FormParams <> '') then
    Result := ARequestInfo.FormParams;
end;

procedure TPrismRestServer.SendJson(AResponseInfo: TIdHTTPResponseInfo;
  Obj: TJSONObject; Code: Integer);
begin
  try
    AResponseInfo.ResponseNo := Code;
    AResponseInfo.ContentType := 'application/json';
    AResponseInfo.CharSet := 'utf-8';
    AResponseInfo.ContentText := Obj.ToJSON;
  finally
    Obj.Free;
  end;
end;

procedure TPrismRestServer.SendError(AResponseInfo: TIdHTTPResponseInfo;
  Code: Integer; const Msg: string);
var
  Root, Err: TJSONObject;
begin
  Root := TJSONObject.Create;
  Err := TJSONObject.Create;
  Err.AddPair('message', Msg);
  Err.AddPair('type', 'invalid_request_error');
  Root.AddPair('error', Err);
  SendJson(AResponseInfo, Root, Code);
end;

procedure TPrismRestServer.BeginStream(AResponseInfo: TIdHTTPResponseInfo;
  const ContentType: string);
begin
  AResponseInfo.ResponseNo := 200;
  AResponseInfo.ContentType := ContentType;
  AResponseInfo.CharSet := 'utf-8';
  { Chunked: otherwise Indy's WriteHeader forces a Content-Length and the
    client disconnects after the first segment }
  AResponseInfo.TransferEncoding := 'chunked';
  AResponseInfo.CloseConnection := True;
  AResponseInfo.CustomHeaders.AddValue('Cache-Control', 'no-cache');
  AResponseInfo.WriteHeader;
end;

procedure TPrismRestServer.StreamChunk(AContext: TIdContext; const S: string);
var
  B: TIdBytes;
begin
  B := ToBytes(S, IndyTextEncoding_UTF8);
  AContext.Connection.IOHandler.WriteLn(IntToHex(Length(B), 1));
  AContext.Connection.IOHandler.Write(B);
  AContext.Connection.IOHandler.WriteLn('');
end;

procedure TPrismRestServer.StreamEnd(AContext: TIdContext);
begin
  AContext.Connection.IOHandler.WriteLn('0');
  AContext.Connection.IOHandler.WriteLn('');
end;

procedure TPrismRestServer.ParseMessages(Body: TJSONObject;
  out Msgs: TChatMessages; out LastUser: string);
var
  Arr: TJSONArray;
  I: Integer;
  M: TJSONObject;
begin
  SetLength(Msgs, 0);
  LastUser := '';
  Arr := Body.GetValue('messages') as TJSONArray;
  if Arr = nil then
    Exit;
  SetLength(Msgs, Arr.Count);
  for I := 0 to Arr.Count - 1 do
  begin
    M := Arr.Items[I] as TJSONObject;
    Msgs[I] := TChatMessage.Make(GetStr(M, 'role', 'user'),
      GetStr(M, 'content', ''));
    if SameText(Msgs[I].Role, 'user') then
      LastUser := Msgs[I].Content;
  end;
end;

function TPrismRestServer.ParseSampling(Body: TJSONObject): TSamplingParams;
var
  OllamaOpts: TJSONObject;
begin
  Result := TSamplingParams.Default;
  Result.Temperature := GetNum(Body, 'temperature', Result.Temperature);
  Result.TopP := GetNum(Body, 'top_p', Result.TopP);
  Result.TopK := Round(GetNum(Body, 'top_k', Result.TopK));
  Result.MaxTokens := Round(GetNum(Body, 'max_tokens',
    GetNum(Body, 'max_completion_tokens', Result.MaxTokens)));
  Result.Seed := UInt64(Round(GetNum(Body, 'seed', 0)));
  { Ollama packs options into "options" }
  OllamaOpts := Body.GetValue('options') as TJSONObject;
  if OllamaOpts <> nil then
  begin
    Result.Temperature := GetNum(OllamaOpts, 'temperature', Result.Temperature);
    Result.TopP := GetNum(OllamaOpts, 'top_p', Result.TopP);
    Result.TopK := Round(GetNum(OllamaOpts, 'top_k', Result.TopK));
    Result.MaxTokens := Round(GetNum(OllamaOpts, 'num_predict',
      Result.MaxTokens));
  end;
end;

procedure TPrismRestServer.HandleChat(AContext: TIdContext;
  Body: TJSONObject; AResponseInfo: TIdHTTPResponseInfo; OllamaStyle: Boolean);
const
  TOOLS_PROMPT =
    'You have access to an exact calculator tool. Whenever you need to ' +
    'compute something numeric, write <<calc: EXPRESSION>> and stop; the ' +
    'exact result will be inserted as <<result: VALUE>>, then continue ' +
    'your answer using that value. Supported: + - * / ^ ( ), functions ' +
    'sqrt, sin, cos, tan, exp, ln, log, abs, min, max, pow(a,b), round, ' +
    'and constants pi, e, c, g, h, kb, na, qe, r, ke. ' +
    'Example: 12.5*380 is <<calc: 12.5*380>> <<result: 4750>> 4750.';
var
  Msgs, MsgsWithTools: TChatMessages;
  LastUser, Id, Text: string;
  SP: TSamplingParams;
  Stream, DoVerify, UseTools: Boolean;
  PromptTokens: TArray<Integer>;
  Gen: TGenerator;
  Usage: TUsage;
  Root, Choice, Msg, UsageObj: TJSONObject;
  Choices, Areas: TJSONArray;
  Ver: TVerificationResult;
  Created: Int64;
  StreamWrite: TProc<string>;
  ChunkJson: TFunc<string, Boolean, string>;
  ToolFn: TToolHandler;
  AreaHist: TArray<Int64>;
  I: Integer;
begin
  ToolFn :=
    function(const Call: string): string
    var
      V: Double;
      Err: string;
    begin
      if TryEvalExpression(Call, nil, V, Err) then
        Result := FormatValue(V)
      else
        Result := 'ERROR: ' + Err;
    end;
  { As closure variables instead of local routines, so they can be called
    from within the OnToken callback (E2555) }
  StreamWrite :=
    procedure(S: string)
    begin
      if OllamaStyle then
        StreamChunk(AContext, S + #10)             // NDJSON
      else
        StreamChunk(AContext, 'data: ' + S + #10#10); // SSE
    end;
  ChunkJson :=
    function(Content: string; Final: Boolean): string
    var
      C, Ch, D, M: TJSONObject;
      ChArr: TJSONArray;
    begin
      C := TJSONObject.Create;
      try
        if OllamaStyle then
        begin
          C.AddPair('model', FBackend.ModelName);
          C.AddPair('created_at', IsoNowUtc);
          M := TJSONObject.Create;
          M.AddPair('role', 'assistant');
          M.AddPair('content', Content);
          C.AddPair('message', M);
          C.AddPair('done', TJSONBool.Create(Final));
          { Erst im Schlusspaket -- vorher stehen die Zahlen noch nicht fest.
            Ollama macht es genauso, Clients erwarten sie dort. }
          if Final then
          begin
            C.AddPair('prompt_eval_count',
              TJSONNumber.Create(Usage.PromptTokens));
            C.AddPair('eval_count',
              TJSONNumber.Create(Usage.CompletionTokens));
            AddTimings(C, Usage, True);
          end;
        end
        else
        begin
          C.AddPair('id', Id);
          C.AddPair('object', 'chat.completion.chunk');
          C.AddPair('created', TJSONNumber.Create(Created));
          C.AddPair('model', FBackend.ModelName);
          ChArr := TJSONArray.Create;
          Ch := TJSONObject.Create;
          Ch.AddPair('index', TJSONNumber.Create(0));
          D := TJSONObject.Create;
          if not Final then
            D.AddPair('content', Content);
          Ch.AddPair('delta', D);
          if Final then
            Ch.AddPair('finish_reason', 'stop')
          else
            Ch.AddPair('finish_reason', TJSONNull.Create);
          ChArr.AddElement(Ch);
          C.AddPair('choices', ChArr);
          { OpenAI kennt fuer den Strom kein Zeitfeld -- die Erweiterung reist
            im Schlusschunk mit, damit ein Stromverbraucher dieselbe Auskunft
            bekommt wie einer ohne Strom. }
          if Final then
            AddTimings(C, Usage, False);
        end;
        Result := C.ToJSON;
      finally
        C.Free;
      end;
    end;

  ParseMessages(Body, Msgs, LastUser);
  if Length(Msgs) = 0 then
  begin
    SendError(AResponseInfo, 400, 'Field "messages" is missing or empty.');
    Exit;
  end;
  SP := ParseSampling(Body);
  Stream := GetBool(Body, 'stream', OllamaStyle);
  DoVerify := GetBool(Body, 'verify', FOpts.VerifyDefault) and not Stream;
  UseTools := GetBool(Body, 'use_tools', False) or
    (Body.GetValue('tools') <> nil);
  if UseTools and (FBackend is TLlamaBackend) then
  begin
    { GGUF instruct models: prepend a system message that teaches the
      <<calc: ...>> protocol. Native Prism models are expected to have
      learned the protocol from their training corpus instead - an
      unseen system text would only derail a small model. }
    SetLength(MsgsWithTools, Length(Msgs) + 1);
    MsgsWithTools[0] := TChatMessage.Make('system', TOOLS_PROMPT);
    for I := 0 to High(Msgs) do
      MsgsWithTools[I + 1] := Msgs[I];
    Msgs := MsgsWithTools;
  end;
  PromptTokens := FBackend.Tokenizer.BuildChatTokens(Msgs,
    FTemplate);
  Id := NewId('chatcmpl-');
  Created := UnixNow;
  AreaHist := nil;

  FGenLock.Enter;
  try
    Gen := SharedGenerator;
    if Stream then
    begin
      if OllamaStyle then
        BeginStream(AResponseInfo, 'application/x-ndjson')
      else
        BeginStream(AResponseInfo, 'text/event-stream');
      if UseTools then
        Gen.GenerateWithTools(PromptTokens, SP,
          procedure(Chunk: string)
          begin
            StreamWrite(ChunkJson(Chunk, False));
          end, Usage, ToolFn)
      else
        Gen.Generate(PromptTokens, SP,
          procedure(Chunk: string)
          begin
            StreamWrite(ChunkJson(Chunk, False));
          end, Usage);
      StreamWrite(ChunkJson('', True));
      if not OllamaStyle then
        StreamWrite('[DONE]');
      StreamEnd(AContext);
      Exit;
    end;
    if UseTools then
      Text := Gen.GenerateWithTools(PromptTokens, SP, nil, Usage, ToolFn)
    else
      Text := Gen.Generate(PromptTokens, SP, nil, Usage);
    { thematic areas: MoE routing histogram of this request }
    if (Gen.Engine is TPrismEngine) and
      (Length(TPrismEngine(Gen.Engine).ExpertHistogram) > 1) then
      AreaHist := Copy(TPrismEngine(Gen.Engine).ExpertHistogram);

    if DoVerify then
      Ver := FVerifier.Verify(PromptTokens, LastUser, Text);
  finally
    FGenLock.Leave;
  end;

  Root := TJSONObject.Create;
  if OllamaStyle then
  begin
    Root.AddPair('model', FBackend.ModelName);
    Root.AddPair('created_at', IsoNowUtc);
    Msg := TJSONObject.Create;
    Msg.AddPair('role', 'assistant');
    Msg.AddPair('content', Text);
    Root.AddPair('message', Msg);
    Root.AddPair('done', TJSONBool.Create(True));
    Root.AddPair('prompt_eval_count', TJSONNumber.Create(Usage.PromptTokens));
    Root.AddPair('eval_count', TJSONNumber.Create(Usage.CompletionTokens));
    AddTimings(Root, Usage, True);
  end
  else
  begin
    Root.AddPair('id', Id);
    Root.AddPair('object', 'chat.completion');
    Root.AddPair('created', TJSONNumber.Create(Created));
    Root.AddPair('model', FBackend.ModelName);
    Choices := TJSONArray.Create;
    Choice := TJSONObject.Create;
    Choice.AddPair('index', TJSONNumber.Create(0));
    Msg := TJSONObject.Create;
    Msg.AddPair('role', 'assistant');
    Msg.AddPair('content', Text);
    Choice.AddPair('message', Msg);
    Choice.AddPair('finish_reason', 'stop');
    Choices.AddElement(Choice);
    Root.AddPair('choices', Choices);
    UsageObj := TJSONObject.Create;
    UsageObj.AddPair('prompt_tokens', TJSONNumber.Create(Usage.PromptTokens));
    UsageObj.AddPair('completion_tokens',
      TJSONNumber.Create(Usage.CompletionTokens));
    UsageObj.AddPair('total_tokens',
      TJSONNumber.Create(Usage.PromptTokens + Usage.CompletionTokens));
    Root.AddPair('usage', UsageObj);
    AddTimings(Root, Usage, False);
  end;
  if DoVerify then
    Root.AddPair('x_verification', Ver.ToJson);
  if Length(AreaHist) > 1 then
  begin
    { router decisions per expert ("thematic area") for this request }
    Areas := TJSONArray.Create;
    for I := 0 to High(AreaHist) do
      Areas.Add(AreaHist[I]);
    Root.AddPair('x_areas', Areas);
  end;
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.HandleCompletions(AContext: TIdContext;
  Body: TJSONObject; AResponseInfo: TIdHTTPResponseInfo);
var
  Prompt, Text, Id: string;
  SP: TSamplingParams;
  Tokens: TArray<Integer>;
  Gen: TGenerator;
  Usage: TUsage;
  Root, Choice, UsageObj: TJSONObject;
  Choices: TJSONArray;
  Tok: TLlmTokenizerBase;
begin
  Prompt := GetStr(Body, 'prompt', '');
  SP := ParseSampling(Body);
  Tok := FBackend.Tokenizer;
  if Tok.PrependBos then
    Tokens := [Tok.BosId] + Tok.Encode(Prompt)
  else
    Tokens := Tok.Encode(Prompt);
  Id := NewId('cmpl-');
  FGenLock.Enter;
  try
    Gen := SharedGenerator;
    Text := Gen.Generate(Tokens, SP, nil, Usage);
  finally
    FGenLock.Leave;
  end;
  Root := TJSONObject.Create;
  Root.AddPair('id', Id);
  Root.AddPair('object', 'text_completion');
  Root.AddPair('created', TJSONNumber.Create(UnixNow));
  Root.AddPair('model', FBackend.ModelName);
  Choices := TJSONArray.Create;
  Choice := TJSONObject.Create;
  Choice.AddPair('index', TJSONNumber.Create(0));
  Choice.AddPair('text', Text);
  Choice.AddPair('finish_reason', 'stop');
  Choices.AddElement(Choice);
  Root.AddPair('choices', Choices);
  UsageObj := TJSONObject.Create;
  UsageObj.AddPair('prompt_tokens', TJSONNumber.Create(Usage.PromptTokens));
  UsageObj.AddPair('completion_tokens',
    TJSONNumber.Create(Usage.CompletionTokens));
  UsageObj.AddPair('total_tokens',
    TJSONNumber.Create(Usage.PromptTokens + Usage.CompletionTokens));
  Root.AddPair('usage', UsageObj);
  AddTimings(Root, Usage, False);
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.HandleGenerate(AContext: TIdContext;
  Body: TJSONObject; AResponseInfo: TIdHTTPResponseInfo);
var
  Prompt, Text: string;
  SP: TSamplingParams;
  Stream: Boolean;
  Tokens: TArray<Integer>;
  Gen: TGenerator;
  Usage: TUsage;
  Root: TJSONObject;
  Tok: TLlmTokenizerBase;
  LineJson: TFunc<string, Boolean, string>;
begin
  LineJson :=
    function(Response: string; Done: Boolean): string
    var
      C: TJSONObject;
    begin
      C := TJSONObject.Create;
      try
        C.AddPair('model', FBackend.ModelName);
        C.AddPair('created_at', IsoNowUtc);
        C.AddPair('response', Response);
        C.AddPair('done', TJSONBool.Create(Done));
        if Done then
        begin
          C.AddPair('prompt_eval_count',
            TJSONNumber.Create(Usage.PromptTokens));
          C.AddPair('eval_count', TJSONNumber.Create(Usage.CompletionTokens));
          AddTimings(C, Usage, True);
        end;
        Result := C.ToJSON;
      finally
        C.Free;
      end;
    end;

  Prompt := GetStr(Body, 'prompt', '');
  SP := ParseSampling(Body);
  Stream := GetBool(Body, 'stream', True);
  Tok := FBackend.Tokenizer;
  if Tok.PrependBos then
    Tokens := [Tok.BosId] + Tok.Encode(Prompt)
  else
    Tokens := Tok.Encode(Prompt);
  FGenLock.Enter;
  try
    Gen := SharedGenerator;
    if Stream then
    begin
      BeginStream(AResponseInfo, 'application/x-ndjson');
      Gen.Generate(Tokens, SP,
        procedure(Chunk: string)
        begin
          StreamChunk(AContext, LineJson(Chunk, False) + #10);
        end, Usage);
      StreamChunk(AContext, LineJson('', True) + #10);
      StreamEnd(AContext);
      Exit;
    end;
    Text := Gen.Generate(Tokens, SP, nil, Usage);
  finally
    FGenLock.Leave;
  end;
  Root := TJSONObject.Create;
  Root.AddPair('model', FBackend.ModelName);
  Root.AddPair('created_at', IsoNowUtc);
  Root.AddPair('response', Text);
  Root.AddPair('done', TJSONBool.Create(True));
  Root.AddPair('prompt_eval_count', TJSONNumber.Create(Usage.PromptTokens));
  Root.AddPair('eval_count', TJSONNumber.Create(Usage.CompletionTokens));
  AddTimings(Root, Usage, True);
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.HandleTrain(Body: TJSONObject;
  AResponseInfo: TIdHTTPResponseInfo);
var
  UserT, AssistantT, TextT, DataB64, ModalityS, Desc: string;
  Sample, Raw: TBytes;
  Root: TJSONObject;
begin
  UserT := GetStr(Body, 'user', '');
  AssistantT := GetStr(Body, 'assistant', '');
  TextT := GetStr(Body, 'text', '');
  DataB64 := GetStr(Body, 'data', '');
  ModalityS := GetStr(Body, 'modality', 'text');
  Desc := GetStr(Body, 'description', '');

  if DataB64 <> '' then
  begin
    Raw := TNetEncoding.Base64.DecodeStringToBytes(DataB64);
    Sample := MakeDataSample(Raw, ModalityFromString(ModalityS), Desc);
  end
  else if (UserT <> '') and (AssistantT <> '') then
    Sample := MakeTextSample(UserT, AssistantT)
  else if TextT <> '' then
    Sample := TEncoding.UTF8.GetBytes(TextT + '<|eos|>'#10)
  else
  begin
    SendError(AResponseInfo, 400,
      'Expected: {"user","assistant"} or {"text"} or {"data","modality","description"}.');
    Exit;
  end;

  AppendToCorpus(FOpts.CorpusPath, Sample);
  Root := TJSONObject.Create;
  Root.AddPair('sample_bytes', TJSONNumber.Create(Length(Sample)));
  Root.AddPair('corpus', FOpts.CorpusPath);
  if FTrainSvc <> nil then
  begin
    FTrainSvc.EnqueueSample(Sample);
    Root.AddPair('status', 'training'); // online finetuning runs in the background
  end
  else
    Root.AddPair('status', 'stored');   // offline training via the PrismTrain CLI
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.HandleModels(AResponseInfo: TIdHTTPResponseInfo);
var
  Root, M: TJSONObject;
  Arr: TJSONArray;
begin
  Root := TJSONObject.Create;
  Root.AddPair('object', 'list');
  Arr := TJSONArray.Create;
  M := TJSONObject.Create;
  M.AddPair('id', FBackend.ModelName);
  M.AddPair('object', 'model');
  M.AddPair('created', TJSONNumber.Create(UnixNow));
  M.AddPair('owned_by', 'prism');
  Arr.AddElement(M);
  Root.AddPair('data', Arr);
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.HandleTags(AResponseInfo: TIdHTTPResponseInfo);
var
  Root, M: TJSONObject;
  Arr: TJSONArray;
begin
  Root := TJSONObject.Create;
  Arr := TJSONArray.Create;
  M := TJSONObject.Create;
  M.AddPair('name', FBackend.ModelName + ':latest');
  M.AddPair('model', FBackend.ModelName + ':latest');
  M.AddPair('modified_at', IsoNowUtc);
  M.AddPair('size', TJSONNumber.Create(0));
  Arr.AddElement(M);
  Root.AddPair('models', Arr);
  SendJson(AResponseInfo, Root);
end;

{ Lazily built, then reused. MUST be called with FGenLock held -- which every
  call site already does, since generations are serialised anyway. The engine
  resets its position at the start of each Generate, so no state leaks between
  requests. }
function TPrismRestServer.SharedGenerator: TGenerator;
begin
  if FGen = nil then
    FGen := TGenerator.Create(FBackend);
  Result := FGen;
end;

procedure TPrismRestServer.HandleHealth(AResponseInfo: TIdHTTPResponseInfo);
var
  Root: TJSONObject;
begin
  Root := TJSONObject.Create;
  Root.AddPair('status', 'ok');
  Root.AddPair('name', 'Prism');
  Root.AddPair('version', PRISM_VERSION);
  Root.AddPair('model', FBackend.ModelName);
  Root.AddPair('backend', Prism.Gpu.Backend.Name);
  Root.AddPair('gpu', Prism.Gpu.GpuDetails);
  Root.AddPair('gpu_quantized', TJSONBool.Create(Prism.Gpu.GpuQuantizedActive));
  Root.AddPair('training', TJSONBool.Create(FTrainSvc <> nil));
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.HandleLawsList(AResponseInfo: TIdHTTPResponseInfo;
  const Query: string);
var
  Root, LJ, VJ: TJSONObject;
  Arr, VArr: TJSONArray;
  Laws: TArray<TLaw>;
  L: TLaw;
  V: TLawVariable;
  Q: string;
begin
  Q := '';
  if Query.StartsWith('q=') then
    Q := TNetEncoding.URL.Decode(Copy(Query, 3, MaxInt));
  Laws := SearchLaws(Q);
  Root := TJSONObject.Create;
  Arr := TJSONArray.Create;
  for L in Laws do
  begin
    LJ := TJSONObject.Create;
    LJ.AddPair('name', L.Name);
    LJ.AddPair('formula', L.Formula);
    LJ.AddPair('output', L.Output);
    LJ.AddPair('description', L.Description);
    VArr := TJSONArray.Create;
    for V in L.Vars do
    begin
      VJ := TJSONObject.Create;
      VJ.AddPair('name', V.Name);
      VJ.AddPair('unit', V.Units);
      VJ.AddPair('meaning', V.Meaning);
      VArr.AddElement(VJ);
    end;
    LJ.AddPair('variables', VArr);
    Arr.AddElement(LJ);
  end;
  Root.AddPair('laws', Arr);
  Root.AddPair('count', TJSONNumber.Create(Length(Laws)));
  SendJson(AResponseInfo, Root);
end;

function ReadVariables(Body: TJSONObject): TDictionary<string, Double>;
var
  VarsObj: TJSONObject;
  P: TJSONPair;
begin
  Result := TDictionary<string, Double>.Create;
  VarsObj := Body.GetValue('variables') as TJSONObject;
  if VarsObj <> nil then
    for P in VarsObj do
      if P.JsonValue is TJSONNumber then
        Result.AddOrSetValue(LowerCase(P.JsonString.Value),
          TJSONNumber(P.JsonValue).AsDouble);
end;

procedure TPrismRestServer.HandleCalc(Body: TJSONObject;
  AResponseInfo: TIdHTTPResponseInfo);
var
  Expr, Err: string;
  Vars: TDictionary<string, Double>;
  V: Double;
  Root: TJSONObject;
begin
  Expr := GetStr(Body, 'expression', '');
  if Expr = '' then
  begin
    SendError(AResponseInfo, 400, 'Field "expression" is missing.');
    Exit;
  end;
  Vars := ReadVariables(Body);
  try
    if not TryEvalExpression(Expr, Vars, V, Err) then
    begin
      SendError(AResponseInfo, 400, Err);
      Exit;
    end;
  finally
    Vars.Free;
  end;
  Root := TJSONObject.Create;
  Root.AddPair('expression', Expr);
  Root.AddPair('result', TJSONNumber.Create(V));
  Root.AddPair('text', FormatValue(V));
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.HandleLawEval(Body: TJSONObject;
  AResponseInfo: TIdHTTPResponseInfo);
var
  Name, Err: string;
  Vars: TDictionary<string, Double>;
  V: Double;
  Law: TLaw;
  Root: TJSONObject;
begin
  Name := GetStr(Body, 'law', '');
  if Name = '' then
  begin
    SendError(AResponseInfo, 400, 'Field "law" is missing.');
    Exit;
  end;
  Vars := ReadVariables(Body);
  try
    if not TryEvalLaw(Name, Vars, V, Law, Err) then
    begin
      SendError(AResponseInfo, 400, Err);
      Exit;
    end;
  finally
    Vars.Free;
  end;
  Root := TJSONObject.Create;
  Root.AddPair('law', Law.Name);
  Root.AddPair('formula', Law.Formula);
  Root.AddPair('output', Law.Output);
  Root.AddPair('result', TJSONNumber.Create(V));
  Root.AddPair('text', FormatValue(V));
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.DoCommandOther(AContext: TIdContext;
  ARequestInfo: TIdHTTPRequestInfo; AResponseInfo: TIdHTTPResponseInfo);
begin
  { CORS preflight }
  AResponseInfo.ResponseNo := 204;
  AResponseInfo.CustomHeaders.AddValue('Access-Control-Allow-Origin', '*');
  AResponseInfo.CustomHeaders.AddValue('Access-Control-Allow-Methods',
    'GET, POST, OPTIONS');
  AResponseInfo.CustomHeaders.AddValue('Access-Control-Allow-Headers',
    'Content-Type, Authorization');
end;

procedure TPrismRestServer.DoCommand(AContext: TIdContext;
  ARequestInfo: TIdHTTPRequestInfo; AResponseInfo: TIdHTTPResponseInfo);
var
  Doc, BodyS: string;
  Body: TJSONObject;
  IsPost: Boolean;
begin
  Doc := LowerCase(ARequestInfo.Document);
  IsPost := ARequestInfo.CommandType = hcPOST;
  AResponseInfo.CustomHeaders.AddValue('Access-Control-Allow-Origin', '*');
  Body := nil;
  try
    try
      if IsPost then
      begin
        BodyS := ReadBody(ARequestInfo);
        if BodyS <> '' then
          Body := TJSONObject.ParseJSONValue(BodyS) as TJSONObject;
        if Body = nil then
        begin
          SendError(AResponseInfo, 400, 'Invalid JSON body.');
          Exit;
        end;
      end;

      if (Doc = '/v1/chat/completions') and IsPost then
        HandleChat(AContext, Body, AResponseInfo, False)
      else if (Doc = '/api/chat') and IsPost then
        HandleChat(AContext, Body, AResponseInfo, True)
      else if (Doc = '/v1/completions') and IsPost then
        HandleCompletions(AContext, Body, AResponseInfo)
      else if (Doc = '/api/generate') and IsPost then
        HandleGenerate(AContext, Body, AResponseInfo)
      else if ((Doc = '/api/train') or (Doc = '/v1/train')) and IsPost then
        HandleTrain(Body, AResponseInfo)
      else if (Doc = '/v1/tools/calc') and IsPost then
        HandleCalc(Body, AResponseInfo)
      else if (Doc = '/v1/laws/eval') and IsPost then
        HandleLawEval(Body, AResponseInfo)
      else if (Doc = '/v1/laws') and not IsPost then
        HandleLawsList(AResponseInfo, ARequestInfo.QueryParams)
      else if (Doc = '/v1/models') and not IsPost then
        HandleModels(AResponseInfo)
      else if (Doc = '/api/tags') and not IsPost then
        HandleTags(AResponseInfo)
      else if (Doc = '/api/version') and not IsPost then
        SendJson(AResponseInfo, TJSONObject.Create
          .AddPair('version', PRISM_VERSION))
      else if (Doc = '/') or (Doc = '/health') then
        HandleHealth(AResponseInfo)
      else
        SendError(AResponseInfo, 404, 'Unknown endpoint: ' + Doc);
    except
      on E: Exception do
      begin
        Log('Error [' + Doc + ']: ' + E.Message);
        if not AResponseInfo.HeaderHasBeenWritten then
          SendError(AResponseInfo, 500, E.Message);
      end;
    end;
  finally
    Body.Free;
  end;
end;

end.
