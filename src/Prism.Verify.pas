unit Prism.Verify;

{ Integrated self-verification of the generated answers.
  Three independent signals are combined:

  1. Perplexity: How "confident" was the model in its own answer?
     (teacher-forcing rescoring of the answer under the prompt)
  2. Self-consistency: K alternative answers are sampled and compared
     with the answer (Jaccard similarity over token bigrams).
     Stable answers = high consistency.
  3. Critic pass: The model is asked whether the answer is correct,
     and the probabilities of "yes" vs. "no" are compared.

  Result: pass / warn / fail + raw values (in the REST response field
  "x_verification"). Thresholds are configurable. }

interface

uses
  System.SysUtils, System.Math, System.JSON, System.Generics.Collections,
  Prism.Types, Prism.Inference;

type
  TVerificationResult = record
    Perplexity: Double;
    SelfConsistency: Double; // 0..1
    CriticScore: Double;     // 0..1 (P("yes"))
    Verdict: string;         // 'pass' | 'warn' | 'fail'
    function ToJson: TJSONObject;
  end;

  TVerifier = class
  private
    FBackend: TLlmBackend;
    function BigramSimilarity(const A, B: TArray<Integer>): Double;
  public
    PplWarn: Double;   // perplexity above this value -> warn
    PplFail: Double;   // perplexity above this value -> fail
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
  AnswerTokens, CriticTokens, YesTok, NoTok: TArray<Integer>;
  SP: TSamplingParams;
  Usage: TUsage;
  AltTokens: TList<Integer>;
  SimSum, LpYes, LpNo: Double;
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
    { 1. Perplexity of the model's own answer }
    if Length(AnswerTokens) > 0 then
      Result.Perplexity := Gen.Perplexity(PromptTokens, AnswerTokens);

    { 2. Self-consistency over alternative samples }
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

    { 3. Critic pass: P("yes") vs. P("no") }
    SetLength(Msgs, 1);
    Msgs[0] := TChatMessage.Make('user',
      'Check the following answer.'#10 +
      'Question: ' + UserText + #10 +
      'Answer: ' + AnswerText + #10 +
      'Is the answer correct and consistent? Reply with yes or no only.');
    CriticTokens := Tok.BuildChatTokens(Msgs, FBackend.DefaultTemplate);
    YesTok := Tok.Encode('yes');
    NoTok := Tok.Encode('no');
    if (Length(YesTok) > 0) and (Length(NoTok) > 0) then
    begin
      LpYes := Gen.ScoreContinuation(CriticTokens, YesTok) / Length(YesTok);
      LpNo := Gen.ScoreContinuation(CriticTokens, NoTok) / Length(NoTok);
      Result.CriticScore := Exp(LpYes) / Max(1e-12, Exp(LpYes) + Exp(LpNo));
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
