unit Prism.Laws;

{ Law layer: exact, symbolic knowledge next to the statistical model.

  1. Expression evaluator (recursive descent): arithmetic, functions,
     physical constants, user variables. Deterministic and exact -
     the neural model proposes, this unit computes.
  2. Law library: a curated set of physical/mathematical formulas with
     variables, units and descriptions. Queryable and evaluable via REST.
  3. Math-claim checker: extracts arithmetic claims ("6 * 7 = 42",
     "6 mal 7 ergibt 42") from generated answers and re-computes them.
     Used by Prism.Verify as a law-grounded verification signal:
     a failed re-computation falsifies an answer deterministically. }

interface

uses
  System.SysUtils, System.Math, System.Character,
  System.Generics.Collections;

type
  ELawError = class(Exception);

  TLawVariable = record
    Name: string;
    Units: string;
    Meaning: string;
  end;

  TLaw = record
    Name: string;         // e.g. 'kinetic_energy'
    Formula: string;      // display form, e.g. 'E = 0.5 * m * v^2'
    Expr: string;         // evaluable right-hand side
    Output: string;       // result variable incl. unit
    Description: string;
    Vars: TArray<TLawVariable>;
  end;

  TMathCheck = record
    Claim: string;      // normalized claim, e.g. '6*7 = 42'
    Expected: Double;   // recomputed value
    Stated: Double;     // value stated in the answer
    Passed: Boolean;
  end;

{ Evaluate an expression; Vars may be nil. Raises ELawError on failure. }
function EvalExpression(const Expr: string;
  Vars: TDictionary<string, Double> = nil): Double;
function TryEvalExpression(const Expr: string;
  Vars: TDictionary<string, Double>; out Value: Double;
  out Err: string): Boolean;

{ Law library access }
function LawCount: Integer;
function GetLaw(Index: Integer): TLaw;
function FindLaw(const Name: string; out Law: TLaw): Boolean;
function SearchLaws(const Query: string): TArray<TLaw>;
function TryEvalLaw(const Name: string; Vars: TDictionary<string, Double>;
  out Value: Double; out Law: TLaw; out Err: string): Boolean;

{ Extract and re-compute arithmetic claims from free text }
function CheckMathClaims(const Text: string): TArray<TMathCheck>;

{ Invariant float formatting for tool results / JSON }
function FormatValue(V: Double): string;

implementation

var
  GConsts: TDictionary<string, Double>;
  GLaws: TList<TLaw>;

function FormatValue(V: Double): string;
begin
  Result := FloatToStrF(V, ffGeneral, 12, 0, TFormatSettings.Invariant);
end;

{ ---------- Expression evaluator ---------- }

type
  TExprParser = record
  private
    S: string;
    P: Integer;
    Vars: TDictionary<string, Double>;
    procedure SkipWs;
    function Ch: Char;
    function ParseExpr: Double;
    function ParseTerm: Double;
    function ParseUnary: Double;
    function ParsePower: Double;
    function ParseAtom: Double;
    function ParseNumber: Double;
    function ParseIdent: string;
    function CallFunc(const Name: string): Double;
  public
    function Eval(const AExpr: string;
      AVars: TDictionary<string, Double>): Double;
  end;

procedure TExprParser.SkipWs;
begin
  while (P <= Length(S)) and (S[P] <= ' ') do
    Inc(P);
end;

function TExprParser.Ch: Char;
begin
  if P <= Length(S) then
    Result := S[P]
  else
    Result := #0;
end;

function TExprParser.Eval(const AExpr: string;
  AVars: TDictionary<string, Double>): Double;
begin
  S := AExpr;
  P := 1;
  Vars := AVars;
  Result := ParseExpr;
  SkipWs;
  if P <= Length(S) then
    raise ELawError.CreateFmt('Unexpected character "%s" at position %d',
      [S[P], P]);
end;

function TExprParser.ParseExpr: Double;
var
  Op: Char;
begin
  Result := ParseTerm;
  while True do
  begin
    SkipWs;
    Op := Ch;
    if (Op = '+') or (Op = '-') then
    begin
      Inc(P);
      if Op = '+' then
        Result := Result + ParseTerm
      else
        Result := Result - ParseTerm;
    end
    else
      Break;
  end;
end;

function TExprParser.ParseTerm: Double;
var
  Op: Char;
  R: Double;
begin
  Result := ParseUnary;
  while True do
  begin
    SkipWs;
    Op := Ch;
    if (Op = '*') or (Op = '/') or (Op = '%') then
    begin
      Inc(P);
      R := ParseUnary;
      case Op of
        '*': Result := Result * R;
        '/':
          begin
            if R = 0 then
              raise ELawError.Create('Division by zero');
            Result := Result / R;
          end;
        '%':
          begin
            if R = 0 then
              raise ELawError.Create('Division by zero');
            Result := Result - Int(Result / R) * R;
          end;
      end;
    end
    else
      Break;
  end;
end;

function TExprParser.ParseUnary: Double;
begin
  SkipWs;
  if Ch = '-' then
  begin
    Inc(P);
    Result := -ParseUnary;
  end
  else if Ch = '+' then
  begin
    Inc(P);
    Result := ParseUnary;
  end
  else
    Result := ParsePower;
end;

function TExprParser.ParsePower: Double;
var
  Base: Double;
begin
  Base := ParseAtom;
  SkipWs;
  if Ch = '^' then
  begin
    Inc(P);
    Result := Power(Base, ParseUnary); // right-associative
  end
  else
    Result := Base;
end;

function TExprParser.ParseAtom: Double;
var
  Name: string;
  V: Double;
begin
  SkipWs;
  if Ch = '(' then
  begin
    Inc(P);
    Result := ParseExpr;
    SkipWs;
    if Ch <> ')' then
      raise ELawError.Create('Missing ")"');
    Inc(P);
  end
  else if CharInSet(Ch, ['0' .. '9', '.']) then
    Result := ParseNumber
  else if CharInSet(Ch, ['a' .. 'z', 'A' .. 'Z', '_']) then
  begin
    Name := ParseIdent;
    SkipWs;
    if Ch = '(' then
      Result := CallFunc(Name)
    else if (Vars <> nil) and Vars.TryGetValue(LowerCase(Name), V) then
      Result := V
    else if GConsts.TryGetValue(LowerCase(Name), V) then
      Result := V
    else
      raise ELawError.CreateFmt('Unknown variable or constant "%s"', [Name]);
  end
  else
    raise ELawError.CreateFmt('Unexpected character "%s" at position %d',
      [Ch, P]);
end;

function TExprParser.ParseNumber: Double;
var
  Start: Integer;
begin
  Start := P;
  while CharInSet(Ch, ['0' .. '9', '.']) do
    Inc(P);
  if CharInSet(Ch, ['e', 'E']) and (P > Start) then
  begin
    Inc(P);
    if CharInSet(Ch, ['+', '-']) then
      Inc(P);
    while CharInSet(Ch, ['0' .. '9']) do
      Inc(P);
  end;
  if not TryStrToFloat(Copy(S, Start, P - Start), Result,
    TFormatSettings.Invariant) then
    raise ELawError.CreateFmt('Invalid number "%s"', [Copy(S, Start, P - Start)]);
end;

function TExprParser.ParseIdent: string;
var
  Start: Integer;
begin
  Start := P;
  while CharInSet(Ch, ['a' .. 'z', 'A' .. 'Z', '0' .. '9', '_']) do
    Inc(P);
  Result := Copy(S, Start, P - Start);
end;

function TExprParser.CallFunc(const Name: string): Double;
var
  Args: TArray<Double>;
  N: string;

  procedure ParseArgs;
  var
    L: TList<Double>;
  begin
    Inc(P); // '('
    L := TList<Double>.Create;
    try
      SkipWs;
      if Ch <> ')' then
      begin
        L.Add(ParseExpr);
        SkipWs;
        while Ch = ',' do
        begin
          Inc(P);
          L.Add(ParseExpr);
          SkipWs;
        end;
      end;
      if Ch <> ')' then
        raise ELawError.CreateFmt('Missing ")" in call to %s', [Name]);
      Inc(P);
      Args := L.ToArray;
    finally
      L.Free;
    end;
  end;

  procedure Need(Count: Integer);
  begin
    if Length(Args) <> Count then
      raise ELawError.CreateFmt('%s expects %d argument(s)', [Name, Count]);
  end;

begin
  ParseArgs;
  N := LowerCase(Name);
  if N = 'sqrt' then begin Need(1); if Args[0] < 0 then raise ELawError.Create('sqrt of negative value'); Result := Sqrt(Args[0]); end
  else if N = 'sin' then begin Need(1); Result := Sin(Args[0]); end
  else if N = 'cos' then begin Need(1); Result := Cos(Args[0]); end
  else if N = 'tan' then begin Need(1); Result := Tan(Args[0]); end
  else if N = 'asin' then begin Need(1); Result := ArcSin(Args[0]); end
  else if N = 'acos' then begin Need(1); Result := ArcCos(Args[0]); end
  else if N = 'atan' then begin Need(1); Result := ArcTan(Args[0]); end
  else if N = 'atan2' then begin Need(2); Result := ArcTan2(Args[0], Args[1]); end
  else if N = 'exp' then begin Need(1); Result := Exp(Args[0]); end
  else if N = 'ln' then begin Need(1); if Args[0] <= 0 then raise ELawError.Create('ln of non-positive value'); Result := Ln(Args[0]); end
  else if N = 'log' then begin Need(1); if Args[0] <= 0 then raise ELawError.Create('log of non-positive value'); Result := Log10(Args[0]); end
  else if N = 'log2' then begin Need(1); if Args[0] <= 0 then raise ELawError.Create('log2 of non-positive value'); Result := Log2(Args[0]); end
  else if N = 'abs' then begin Need(1); Result := Abs(Args[0]); end
  else if N = 'floor' then begin Need(1); Result := Floor(Args[0]); end
  else if N = 'ceil' then begin Need(1); Result := Ceil(Args[0]); end
  else if N = 'round' then begin Need(1); Result := Round(Args[0]); end
  else if N = 'sign' then begin Need(1); Result := Sign(Args[0]); end
  else if N = 'min' then begin Need(2); Result := Min(Args[0], Args[1]); end
  else if N = 'max' then begin Need(2); Result := Max(Args[0], Args[1]); end
  else if N = 'pow' then begin Need(2); Result := Power(Args[0], Args[1]); end
  else if N = 'deg2rad' then begin Need(1); Result := DegToRad(Args[0]); end
  else if N = 'rad2deg' then begin Need(1); Result := RadToDeg(Args[0]); end
  else
    raise ELawError.CreateFmt('Unknown function "%s"', [Name]);
end;

function EvalExpression(const Expr: string;
  Vars: TDictionary<string, Double>): Double;
var
  Parser: TExprParser;
begin
  if Trim(Expr) = '' then
    raise ELawError.Create('Empty expression');
  Result := Parser.Eval(Expr, Vars);
  if IsNan(Result) or IsInfinite(Result) then
    raise ELawError.Create('Expression result is not a finite number');
end;

function TryEvalExpression(const Expr: string;
  Vars: TDictionary<string, Double>; out Value: Double;
  out Err: string): Boolean;
begin
  Err := '';
  Value := 0;
  try
    Value := EvalExpression(Expr, Vars);
    Result := True;
  except
    on E: Exception do
    begin
      Err := E.Message;
      Result := False;
    end;
  end;
end;

{ ---------- Law library ---------- }

procedure AddLaw(const Name, Formula, Expr, Output, Description: string;
  const VarDefs: array of string);
var
  L: TLaw;
  I: Integer;
  Parts: TArray<string>;
begin
  L.Name := Name;
  L.Formula := Formula;
  L.Expr := Expr;
  L.Output := Output;
  L.Description := Description;
  SetLength(L.Vars, Length(VarDefs));
  for I := 0 to High(VarDefs) do
  begin
    // 'name|unit|meaning'
    Parts := VarDefs[I].Split(['|']);
    L.Vars[I].Name := Parts[0];
    if Length(Parts) > 1 then
      L.Vars[I].Units := Parts[1];
    if Length(Parts) > 2 then
      L.Vars[I].Meaning := Parts[2];
  end;
  GLaws.Add(L);
end;

procedure BuildLaws;
begin
  AddLaw('kinetic_energy', 'E = 0.5 * m * v^2', '0.5 * m * v^2', 'E [J]',
    'Kinetic energy of a moving mass',
    ['m|kg|mass', 'v|m/s|velocity']);
  AddLaw('potential_energy', 'E = m * g * h', 'm * g * h', 'E [J]',
    'Gravitational potential energy near the surface of Earth',
    ['m|kg|mass', 'h|m|height']);
  AddLaw('newton_second', 'F = m * a', 'm * a', 'F [N]',
    'Newton''s second law: force equals mass times acceleration',
    ['m|kg|mass', 'a|m/s^2|acceleration']);
  AddLaw('momentum', 'p = m * v', 'm * v', 'p [kg*m/s]',
    'Linear momentum',
    ['m|kg|mass', 'v|m/s|velocity']);
  AddLaw('gravitation', 'F = G * m1 * m2 / r^2', 'G * m1 * m2 / r^2', 'F [N]',
    'Newton''s law of universal gravitation',
    ['m1|kg|first mass', 'm2|kg|second mass', 'r|m|distance']);
  AddLaw('coulomb', 'F = ke * q1 * q2 / r^2', 'ke * q1 * q2 / r^2', 'F [N]',
    'Coulomb''s law: electrostatic force between two point charges',
    ['q1|C|first charge', 'q2|C|second charge', 'r|m|distance']);
  AddLaw('ohms_law', 'U = R * I', 'R * I', 'U [V]',
    'Ohm''s law: voltage across a resistor (pass R as variable)',
    ['R|Ohm|resistance', 'I|A|current']);
  AddLaw('electric_power', 'P = U * I', 'U * I', 'P [W]',
    'Electric power from voltage and current',
    ['U|V|voltage', 'I|A|current']);
  AddLaw('ideal_gas_pressure', 'p = n * R * T / V', 'n * R * T / V', 'p [Pa]',
    'Ideal gas law solved for pressure (R = gas constant)',
    ['n|mol|amount of substance', 'T|K|temperature', 'V|m^3|volume']);
  AddLaw('free_fall_distance', 's = 0.5 * g * t^2', '0.5 * g * t^2', 's [m]',
    'Distance covered in free fall from rest',
    ['t|s|time']);
  AddLaw('free_fall_velocity', 'v = g * t', 'g * t', 'v [m/s]',
    'Velocity reached in free fall from rest',
    ['t|s|time']);
  AddLaw('uniform_velocity', 'v = s / t', 's / t', 'v [m/s]',
    'Average velocity from distance and time',
    ['s|m|distance', 't|s|time']);
  AddLaw('acceleration', 'a = (v2 - v1) / t', '(v2 - v1) / t', 'a [m/s^2]',
    'Average acceleration from change of velocity over time',
    ['v1|m/s|initial velocity', 'v2|m/s|final velocity', 't|s|time']);
  AddLaw('density', 'rho = m / V', 'm / V', 'rho [kg/m^3]',
    'Density from mass and volume',
    ['m|kg|mass', 'V|m^3|volume']);
  AddLaw('wave_speed', 'v = f * lambda', 'f * lambda', 'v [m/s]',
    'Wave speed from frequency and wavelength',
    ['f|Hz|frequency', 'lambda|m|wavelength']);
  AddLaw('photon_energy', 'E = h * f', 'h * f', 'E [J]',
    'Photon energy from frequency (Planck)',
    ['f|Hz|frequency']);
  AddLaw('mass_energy', 'E = m * c^2', 'm * c^2', 'E [J]',
    'Mass-energy equivalence (Einstein)',
    ['m|kg|mass']);
  AddLaw('pendulum_period', 'T = 2 * pi * sqrt(L / g)',
    '2 * pi * sqrt(L / g)', 'T [s]',
    'Period of a mathematical pendulum (small angles)',
    ['L|m|pendulum length']);
  AddLaw('circle_area', 'A = pi * r^2', 'pi * r^2', 'A [m^2]',
    'Area of a circle',
    ['r|m|radius']);
  AddLaw('sphere_volume', 'V = 4/3 * pi * r^3', '4 / 3 * pi * r^3',
    'V [m^3]', 'Volume of a sphere',
    ['r|m|radius']);
  AddLaw('pythagoras', 'c = sqrt(a^2 + b^2)', 'sqrt(a^2 + b^2)', 'c',
    'Pythagorean theorem: hypotenuse of a right triangle',
    ['a||first leg', 'b||second leg']);
  AddLaw('percent_change', 'pct = (new - old) / old * 100',
    '(new - old) / old * 100', 'pct [%]',
    'Relative change in percent',
    ['old||old value', 'new||new value']);
end;

function LawCount: Integer;
begin
  Result := GLaws.Count;
end;

function GetLaw(Index: Integer): TLaw;
begin
  Result := GLaws[Index];
end;

function FindLaw(const Name: string; out Law: TLaw): Boolean;
var
  L: TLaw;
begin
  for L in GLaws do
    if SameText(L.Name, Name) then
    begin
      Law := L;
      Exit(True);
    end;
  Result := False;
end;

function SearchLaws(const Query: string): TArray<TLaw>;
var
  L: TLaw;
  Res: TList<TLaw>;
  Q: string;
begin
  Q := LowerCase(Trim(Query));
  Res := TList<TLaw>.Create;
  try
    for L in GLaws do
      if (Q = '') or L.Name.ToLower.Contains(Q) or
        L.Description.ToLower.Contains(Q) or L.Formula.ToLower.Contains(Q) then
        Res.Add(L);
    Result := Res.ToArray;
  finally
    Res.Free;
  end;
end;

function TryEvalLaw(const Name: string; Vars: TDictionary<string, Double>;
  out Value: Double; out Law: TLaw; out Err: string): Boolean;
begin
  Value := 0;
  if not FindLaw(Name, Law) then
  begin
    Err := 'Unknown law: ' + Name;
    Exit(False);
  end;
  Result := TryEvalExpression(Law.Expr, Vars, Value, Err);
end;

{ ---------- Math-claim checker ---------- }

type
  TClaimTokKind = (ckNum, ckOp, ckEq, ckNoise);

  TClaimTok = record
    Kind: TClaimTokKind;
    Num: Double;
    NumText: string; // normalized ('.' decimal separator)
    Op: Char;
  end;

function TokenizeClaims(const Text: string): TArray<TClaimTok>;
var
  Toks: TList<TClaimTok>;
  I, N, Start: Integer;
  T: TClaimTok;
  W, NumS: string;
  C: Char;

  procedure AddOp(AOp: Char);
  var
    K: TClaimTok;
  begin
    K.Kind := ckOp;
    K.Op := AOp;
    Toks.Add(K);
  end;

  procedure AddEq;
  var
    K: TClaimTok;
  begin
    K.Kind := ckEq;
    Toks.Add(K);
  end;

  procedure AddNoise;
  var
    K: TClaimTok;
  begin
    K.Kind := ckNoise;
    Toks.Add(K);
  end;

begin
  Toks := TList<TClaimTok>.Create;
  try
    N := Length(Text);
    I := 1;
    while I <= N do
    begin
      C := Text[I];
      if CharInSet(C, ['0' .. '9']) then
      begin
        Start := I;
        while (I <= N) and CharInSet(Text[I], ['0' .. '9', '.', ',']) do
          Inc(I);
        NumS := Copy(Text, Start, I - Start);
        // strip sentence punctuation at the end
        while (NumS <> '') and CharInSet(NumS[Length(NumS)], ['.', ',']) do
          SetLength(NumS, Length(NumS) - 1);
        NumS := StringReplace(NumS, ',', '.', [rfReplaceAll]);
        T.Kind := ckNum;
        T.NumText := NumS;
        if (NumS <> '') and
          TryStrToFloat(NumS, T.Num, TFormatSettings.Invariant) then
          Toks.Add(T)
        else
          AddNoise;
      end
      else if CharInSet(C, ['+', '-', '*', '/', '^', ':']) then
      begin
        if C = ':' then
          AddOp('/')
        else
          AddOp(C);
        Inc(I);
      end
      else if (C = '=') then
      begin
        AddEq;
        Inc(I);
      end
      else if C.IsLetter then
      begin
        Start := I;
        while (I <= N) and Text[I].IsLetter do
          Inc(I);
        W := LowerCase(Copy(Text, Start, I - Start));
        if (W = 'plus') then
          AddOp('+')
        else if (W = 'minus') then
          AddOp('-')
        else if (W = 'mal') or (W = 'times') then
          AddOp('*')
        else if (W = 'durch') or (W = 'by') then
          AddOp('/') // 'geteilt durch' / 'divided by': the first word is noise
        else if (W = 'hoch') then
          AddOp('^')
        else if (W = 'ergibt') or (W = 'equals') or (W = 'ist') or
          (W = 'is') or (W = 'macht') or (W = 'sind') or (W = 'are') or
          (W = 'gleich') then
          AddEq
        else if (W = 'geteilt') or (W = 'divided') or (W = 'x') then
        begin
          if W = 'x' then
            AddOp('*')
          else
            AddNoise;
        end
        else
          AddNoise;
      end
      else
      begin
        if C > ' ' then
          AddNoise
        else
          ; // whitespace: no token
        Inc(I);
      end;
    end;
    Result := Toks.ToArray;
  finally
    Toks.Free;
  end;
end;

function CheckMathClaims(const Text: string): TArray<TMathCheck>;
const
  MAX_CHECKS = 16;
var
  Toks: TArray<TClaimTok>;
  Res: TList<TMathCheck>;
  I, J, K, D, Ops: Integer;
  Expr, StatedText: string;
  Stated, Expected, Tol: Double;
  Neg: Boolean;
  Chk: TMathCheck;
  Err: string;
begin
  Toks := TokenizeClaims(Text);
  Res := TList<TMathCheck>.Create;
  try
    I := 0;
    while (I <= High(Toks)) and (Res.Count < MAX_CHECKS) do
    begin
      if Toks[I].Kind = ckNum then
      begin
        // match: num (op num)+ eq [eq] [-] num
        Expr := Toks[I].NumText;
        Ops := 0;
        J := I + 1;
        while (J + 1 <= High(Toks)) and (Toks[J].Kind = ckOp) and
          (Toks[J + 1].Kind = ckNum) do
        begin
          Expr := Expr + ' ' + Toks[J].Op + ' ' + Toks[J + 1].NumText;
          Inc(Ops);
          Inc(J, 2);
        end;
        if (Ops >= 1) and (J <= High(Toks)) and (Toks[J].Kind = ckEq) then
        begin
          K := J + 1;
          // tolerate 'ist gleich' / 'is equal' style double equality words
          if (K <= High(Toks)) and (Toks[K].Kind = ckEq) then
            Inc(K);
          Neg := False;
          if (K <= High(Toks)) and (Toks[K].Kind = ckOp) and
            (Toks[K].Op = '-') then
          begin
            Neg := True;
            Inc(K);
          end;
          if (K <= High(Toks)) and (Toks[K].Kind = ckNum) then
          begin
            Stated := Toks[K].Num;
            StatedText := Toks[K].NumText;
            if Neg then
              Stated := -Stated;
            if TryEvalExpression(Expr, nil, Expected, Err) then
            begin
              // tolerance: half a unit of the last stated decimal place
              D := 0;
              if Pos('.', StatedText) > 0 then
                D := Length(StatedText) - Pos('.', StatedText);
              Tol := 0.5 * Power(10, -D) + Abs(Expected) * 1e-9 + 1e-12;
              Chk.Claim := Expr + ' = ' + StatedText;
              if Neg then
                Chk.Claim := Expr + ' = -' + StatedText;
              Chk.Expected := Expected;
              Chk.Stated := Stated;
              Chk.Passed := Abs(Expected - Stated) <= Tol;
              Res.Add(Chk);
            end;
            I := K + 1;
            Continue;
          end;
        end;
        I := J;
      end
      else
        Inc(I);
    end;
    Result := Res.ToArray;
  finally
    Res.Free;
  end;
end;

procedure BuildConsts;
begin
  GConsts.Add('pi', Pi);
  GConsts.Add('e', Exp(1.0));
  GConsts.Add('c', 2.99792458e8);        // speed of light [m/s]
  GConsts.Add('g', 9.80665);             // standard gravity [m/s^2]
  GConsts.Add('bigg', 6.67430e-11);      // gravitational constant (alias)
  GConsts.Add('h', 6.62607015e-34);      // Planck constant [J*s]
  GConsts.Add('hbar', 1.054571817e-34);
  GConsts.Add('kb', 1.380649e-23);       // Boltzmann [J/K]
  GConsts.Add('na', 6.02214076e23);      // Avogadro [1/mol]
  GConsts.Add('qe', 1.602176634e-19);    // elementary charge [C]
  GConsts.Add('r', 8.314462618);         // gas constant [J/(mol*K)]
  GConsts.Add('ke', 8.9875517923e9);     // Coulomb constant [N*m^2/C^2]
  GConsts.Add('eps0', 8.8541878128e-12);
  GConsts.Add('mu0', 1.25663706212e-6);
  GConsts.Add('me', 9.1093837015e-31);   // electron mass [kg]
  GConsts.Add('mp', 1.67262192369e-27);  // proton mass [kg]
  GConsts.Add('atm', 101325.0);          // standard atmosphere [Pa]
end;

initialization
  GConsts := TDictionary<string, Double>.Create;
  GLaws := TList<TLaw>.Create;
  BuildConsts;
  BuildLaws;
  { 'G' would collide with 'g' (case-insensitive lookup); register the
    gravitational constant under its own key and map 'G' in law
    expressions explicitly }
  GConsts.Add('g_const', 6.67430e-11);

finalization
  GConsts.Free;
  GLaws.Free;

end.
