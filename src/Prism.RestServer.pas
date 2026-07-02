unit Prism.RestServer;

{ REST-Schnittstelle, kompatibel zu OpenAI- und Ollama-Clients.

  OpenAI-kompatibel:
    GET  /v1/models
    POST /v1/chat/completions   (stream: SSE "data: ...")
    POST /v1/completions
  Ollama-kompatibel:
    GET  /api/tags, /api/version
    POST /api/generate, /api/chat  (stream: NDJSON)
  Prism-Erweiterungen:
    GET  /health                 Status + Modellinfo
    POST /api/train              Trainingssample einspeisen (Text/multimodal)

  Zusatzfelder in Chat-Requests:
    "verify": true  -> Selbst-Verifikation, Ergebnis in "x_verification"

  Implementiert mit TIdHTTPServer (Indy, Bestandteil von Delphi). }

interface

uses
  System.SysUtils, System.Classes, System.SyncObjs, System.JSON,
  System.DateUtils, System.NetEncoding, System.IOUtils,
  System.Generics.Collections,
  IdHTTPServer, IdCustomHTTPServer, IdContext, IdGlobal,
  Prism.Types, Prism.Model, Prism.Streaming, Prism.Tokenizer,
  Prism.Inference, Prism.Llama, Prism.Verify, Prism.Multimodal,
  Prism.Train, Prism.Gpu;

type
  TServerOptions = record
    Port: Integer;
    ModelPath: string;       // .prism oder .gguf
    TokenizerPath: string;   // nur fuer .prism
    StreamLayers: Integer;   // 0 = Modell komplett in den RAM
    MaxCachedExperts: Integer;
    CtxOverride: Integer;    // GGUF-Kontextfenster kappen (RAM)
    VerifyDefault: Boolean;
    TrainEnabled: Boolean;
    CorpusPath: string;
    Template: string;        // auto|prism|chatml|llama2|plain
    UseGpu: Boolean;
    class function Default: TServerOptions; static;
  end;

  TPrismRestServer = class
  private
    FOpts: TServerOptions;
    FHttp: TIdHTTPServer;
    FBackend: TLlmBackend;
    FVerifier: TVerifier;
    FGenLock: TCriticalSection;
    FTrainSvc: TTrainingService;
    FLog: TProc<string>;
    FTemplate: TChatTemplate;
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
  FTemplate := TemplateFromString(AOpts.Template);

  if AOpts.UseGpu then
  begin
    if TryInitGpuBackend(GpuInfo) then
      Log('GPU-Backend aktiv: ' + GpuInfo)
    else
      Log('GPU nicht verfuegbar (' + GpuInfo + '), nutze CPU.');
  end;

  Ext := LowerCase(ExtractFileExt(AOpts.ModelPath));
  if Ext = '.gguf' then
  begin
    Log('Lade GGUF-Modell: ' + AOpts.ModelPath);
    FBackend := TLlamaBackend.Create(AOpts.ModelPath, AOpts.CtxOverride,
      AOpts.StreamLayers, procedure(S: string) begin Log(S); end);
    if FTemplate <> ctAuto then
      TLlamaBackend(FBackend).Template := FTemplate;
    Log(Format('GGUF bereit: %s (Arch %s, Vokabular %d, Kontext %d)',
      [FBackend.ModelName, TLlamaBackend(FBackend).Model.Gguf.Arch,
       TLlamaBackend(FBackend).Model.Cfg.Vocab,
       TLlamaBackend(FBackend).Model.Cfg.CtxLen]));
  end
  else
  begin
    if not FileExists(AOpts.TokenizerPath) then
      raise Exception.Create('Tokenizer-Datei fehlt: ' + AOpts.TokenizerPath);
    Tok := TTokenizer.Create;
    Tok.LoadFromFile(AOpts.TokenizerPath);
    if AOpts.StreamLayers > 0 then
    begin
      Log(Format('Lade Prism-Modell (Streaming, %d Layer im Cache): %s',
        [AOpts.StreamLayers, AOpts.ModelPath]));
      Prov := TLayerStore.Create(AOpts.ModelPath, AOpts.StreamLayers,
        AOpts.MaxCachedExperts);
    end
    else
    begin
      Log('Lade Prism-Modell (komplett im RAM): ' + AOpts.ModelPath);
      Full := TFullWeights.Create;
      Full.LoadFromFile(AOpts.ModelPath);
      Prov := Full;
    end;
    Log('Konfiguration: ' + Prov.Config.ToString);
    FBackend := TPrismBackend.Create(Prov, Tok,
      TPath.GetFileNameWithoutExtension(AOpts.ModelPath));
    if AOpts.TrainEnabled then
    begin
      if Prov is TFullWeights then
      begin
        FTrainSvc := TTrainingService.Create(TFullWeights(Prov), Tok,
          FGenLock, AOpts.ModelPath, nil,
          procedure(S: string) begin Log(S); end);
        Log('Online-Training aktiv (POST /api/train).');
      end
      else
        Log('Hinweis: Online-Training braucht --stream-layers 0; deaktiviert.');
    end;
  end;

  FVerifier := TVerifier.Create(FBackend);

  FHttp := TIdHTTPServer.Create(nil);
  FHttp.DefaultPort := AOpts.Port;
  FHttp.OnCommandGet := DoCommand;
  FHttp.OnCommandOther := DoCommandOther;
  { Body immer als PostStream durchreichen - sonst konsumiert Indy
    form-kodierte POSTs (Clients ohne Content-Type: application/json) }
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
  Log(Format('Prism-Server laeuft auf Port %d', [FOpts.Port]));
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
  { Clients ohne "Content-Type: application/json": Indy legt den Body bei
    form-urlencoded in FormParams ab statt in PostStream }
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
  { Chunked: sonst erzwingt Indys WriteHeader eine Content-Length und der
    Client trennt nach dem ersten Segment }
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
  { Ollama packt Optionen in "options" }
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
var
  Msgs: TChatMessages;
  LastUser, Id, Text: string;
  SP: TSamplingParams;
  Stream, DoVerify: Boolean;
  PromptTokens: TArray<Integer>;
  Gen: TGenerator;
  Usage: TUsage;
  Root, Choice, Msg, UsageObj: TJSONObject;
  Choices: TJSONArray;
  Ver: TVerificationResult;
  Created: Int64;
  StreamWrite: TProc<string>;
  ChunkJson: TFunc<string, Boolean, string>;
begin
  { Als Closure-Variablen statt lokaler Routinen, damit sie aus dem
    OnToken-Callback heraus aufrufbar sind (E2555) }
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
        end;
        Result := C.ToJSON;
      finally
        C.Free;
      end;
    end;

  ParseMessages(Body, Msgs, LastUser);
  if Length(Msgs) = 0 then
  begin
    SendError(AResponseInfo, 400, 'Feld "messages" fehlt oder ist leer.');
    Exit;
  end;
  SP := ParseSampling(Body);
  Stream := GetBool(Body, 'stream', OllamaStyle);
  DoVerify := GetBool(Body, 'verify', FOpts.VerifyDefault) and not Stream;
  PromptTokens := FBackend.Tokenizer.BuildChatTokens(Msgs,
    FTemplate);
  Id := NewId('chatcmpl-');
  Created := UnixNow;

  FGenLock.Enter;
  try
    Gen := TGenerator.Create(FBackend);
    try
      if Stream then
      begin
        if OllamaStyle then
          BeginStream(AResponseInfo, 'application/x-ndjson')
        else
          BeginStream(AResponseInfo, 'text/event-stream');
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
      Text := Gen.Generate(PromptTokens, SP, nil, Usage);
    finally
      Gen.Free;
    end;

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
  end;
  if DoVerify then
    Root.AddPair('x_verification', Ver.ToJson);
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
    Gen := TGenerator.Create(FBackend);
    try
      Text := Gen.Generate(Tokens, SP, nil, Usage);
    finally
      Gen.Free;
    end;
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
    Gen := TGenerator.Create(FBackend);
    try
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
      Gen.Free;
    end;
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
      'Erwartet: {"user","assistant"} oder {"text"} oder {"data","modality","description"}.');
    Exit;
  end;

  AppendToCorpus(FOpts.CorpusPath, Sample);
  Root := TJSONObject.Create;
  Root.AddPair('sample_bytes', TJSONNumber.Create(Length(Sample)));
  Root.AddPair('corpus', FOpts.CorpusPath);
  if FTrainSvc <> nil then
  begin
    FTrainSvc.EnqueueSample(Sample);
    Root.AddPair('status', 'training'); // Online-Finetuning laeuft im Hintergrund
  end
  else
    Root.AddPair('status', 'stored');   // Offline-Training via PrismTrain-CLI
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
  Root.AddPair('training', TJSONBool.Create(FTrainSvc <> nil));
  SendJson(AResponseInfo, Root);
end;

procedure TPrismRestServer.DoCommandOther(AContext: TIdContext;
  ARequestInfo: TIdHTTPRequestInfo; AResponseInfo: TIdHTTPResponseInfo);
begin
  { CORS-Preflight }
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
          SendError(AResponseInfo, 400, 'Ungueltiger JSON-Body.');
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
        SendError(AResponseInfo, 404, 'Unbekannter Endpunkt: ' + Doc);
    except
      on E: Exception do
      begin
        Log('Fehler [' + Doc + ']: ' + E.Message);
        if not AResponseInfo.HeaderHasBeenWritten then
          SendError(AResponseInfo, 500, E.Message);
      end;
    end;
  finally
    Body.Free;
  end;
end;

end.
