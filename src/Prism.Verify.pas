unit Prism.Verify;

{ Integrierte Selbst-Verifikation der generierten Antworten.
  Drei unabhaengige Signale werden kombiniert:

  1. Perplexitaet: Wie "sicher" war das Modell bei seiner eigenen Antwort?
     (Teacher-Forcing-Rescoring der Antwort unter dem Prompt)
  2. Selbstkonsistenz: K alternative Antworten werden gesampelt und mit
     der Antwort verglichen (Jaccard-Aehnlichkeit ueber Token-Bigramme).
     Stabile Antworten = hohe Konsistenz.
  3. Critic-Pass: Das Modell wird gefragt, ob die Antwort korrekt ist,
     und die Wahrscheinlichkeiten von "ja" vs. "nein" werden verglichen.

  Ergebnis: pass / warn / fail + Rohwerte (im REST-Response-Feld
  "x_verification"). Schwellwerte sind konfigurierbar. }

interface

uses
  System.SysUtils, System.Math, System.JSON, System.Generics.Collections,
  Prism.Types, Prism.Inference;

type
  TVerificationResult = record
    Perplexity: Double;
    SelfConsistency: Double; // 0..1
    CriticScore: Double;     // 0..1 (P("ja"))
    Verdict: string;         // 'pass' | 'warn' | 'fail'
    function ToJson: TJSONObject;
  end;

  TVerifier = class
  private
    FBackend: TLlmBackend;
    function BigramSimilarity(const A, B: TArray<Integer>): Double;
  public
    PplWarn: Double;   // Perplexitaet ueber diesem Wert -> warn
    PplFail: Double;   // Perplexitaet ueber diesem Wert -> fail
    MinConsistency: Double;
    MinCritic: Double;
    constructor Create(ABackend: TLlmBackend);
    function Verify(const PromptTokens: TArray<Integer>;
      const UserText, AnswerText: string;
      Samples: Integer = 2): TVerificationResult;
  end;

implementation

{ TVerificationResult }

function TVerificationResult.ToJson: TJSONObject;
begin
  Result := TJSONObject.Create;
  Result.AddPair('perplexity', TJSONNumber.Create(RoundTo(Perplexity, -3)));
  Result.AddPair('self_consistency',
    TJSONNumber.Create(RoundTo(SelfConsistency, -3)));
  Result.AddPair('critic_score', TJSONNumber.Create(RoundTo(CriticScore, -3)));
  Result.AddPair('verdict', Verdict);
end;

{ TVerifier }

constructor TVerifier.Create(ABackend: TLlmBackend);
begin
  inherited Create;
  FBackend := ABackend;
  PplWarn := 12.0;
  PplFail := 60.0;
  MinConsistency := 0.34;
  MinCritic := 0.45;
end;

function TVerifier.BigramSimilarity(const A, B: TArray<Integer>): Double;
var
  SetA, SetB: TDictionary<Int64, Boolean>;
  I, Inter, Union: Integer;
  K: Int64;
begin
  if (Length(A) < 2) or (Length(B) < 2) then
  begin
    if (Length(A) = Length(B)) and (Length(A) > 0) then
      Exit(1.0)
    else
      Exit(0.0);
  end;
  SetA := TDictionary<Int64, Boolean>.Create;
  SetB := TDictionary<Int64, Boolean>.Create;
  try
    for I := 0 to High(A) - 1 do
      SetA.AddOrSetValue((Int64(A[I]) shl 32) or UInt32(A[I + 1]), True);
    for I := 0 to High(B) - 1 do
      SetB.AddOrSetValue((Int64(B[I]) shl 32) or UInt32(B[I + 1]), True);
    Inter := 0;
    for K in SetA.Keys do
      if SetB.ContainsKey(K) then
        Inc(Inter);
    Union := SetA.Count + SetB.Count - Inter;
    if Union = 0 then
      Result := 0
    else
      Result := Inter / Union;
  finally
    SetA.Free;
    SetB.Free;
  end;
end;

function TVerifier.Verify(const PromptTokens: TArray<Integer>;
  const UserText, AnswerText: string; Samples: Integer): TVerificationResult;
var
  Gen: TGenerator;
  Tok: TLlmTokenizerBase;
  AnswerTokens, CriticTokens, JaTok, NeinTok: TArray<Integer>;
  SP: TSamplingParams;
  Usage: TUsage;
  AltTokens: TList<Integer>;
  SimSum, LpJa, LpNein: Double;
  S: Integer;
  Msgs: TChatMessages;
begin
  Tok := FBackend.Tokenizer;
  AnswerTokens := Tok.Encode(AnswerText);
  Result.Perplexity := 0;
  Result.SelfConsistency := 0;
  Result.CriticScore := 0.5;

  Gen := TGenerator.Create(FBackend);
  try
    { 1. Perplexitaet der eigenen Antwort }
    if Length(AnswerTokens) > 0 then
      Result.Perplexity := Gen.Perplexity(PromptTokens, AnswerTokens);

    { 2. Selbstkonsistenz ueber alternative Samples }
    if Samples > 0 then
    begin
      SP := TSamplingParams.Default;
      SP.Temperature := 0.9;
      SP.MaxTokens := Min(128, Max(16, 2 * Length(AnswerTokens)));
      SimSum := 0;
      for S := 1 to Samples do
      begin
        SP.Seed := UInt64(S) * 7919;
        AltTokens := TList<Integer>.Create;
        try
          Gen.Generate(PromptTokens, SP, nil, Usage, AltTokens);
          SimSum := SimSum + BigramSimilarity(AnswerTokens, AltTokens.ToArray);
        finally
          AltTokens.Free;
        end;
      end;
      Result.SelfConsistency := SimSum / Samples;
    end;

    { 3. Critic-Pass: P("ja") vs. P("nein") }
    SetLength(Msgs, 1);
    Msgs[0] := TChatMessage.Make('user',
      'Pruefe die folgende Antwort.'#10 +
      'Frage: ' + UserText + #10 +
      'Antwort: ' + AnswerText + #10 +
      'Ist die Antwort korrekt und konsistent? Antworte nur mit ja oder nein.');
    CriticTokens := Tok.BuildChatTokens(Msgs, FBackend.DefaultTemplate);
    JaTok := Tok.Encode('ja');
    NeinTok := Tok.Encode('nein');
    if (Length(JaTok) > 0) and (Length(NeinTok) > 0) then
    begin
      LpJa := Gen.ScoreContinuation(CriticTokens, JaTok) / Length(JaTok);
      LpNein := Gen.ScoreContinuation(CriticTokens, NeinTok) / Length(NeinTok);
      Result.CriticScore := Exp(LpJa) / Max(1e-12, Exp(LpJa) + Exp(LpNein));
    end;
  finally
    Gen.Free;
  end;

  if (Result.Perplexity > PplFail) or (Result.CriticScore < MinCritic / 2) then
    Result.Verdict := 'fail'
  else if (Result.Perplexity <= PplWarn) and
    (Result.SelfConsistency >= MinConsistency) and
    (Result.CriticScore >= MinCritic) then
    Result.Verdict := 'pass'
  else
    Result.Verdict := 'warn';
end;

end.
