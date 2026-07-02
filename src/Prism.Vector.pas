unit Prism.Vector;

{ Optimized vector/matrix kernels (CPU), pointer-based.

  Design decisions for performance:
  - PSingle pointer arithmetic instead of array indexing (no range checks,
    better register scheduling by the compiler)
  - 4-way unrolled dot products with double accumulators
  - Row-parallel MatVec via TParallel.For (System.Threading)
  - Quantized fused kernels (Q8_0/Q4_0/Q4_1): the activation is quantized
    to int8 ONCE per call, then pure integer MACs per block. This allows
    GGUF models with billions of parameters to be computed directly in
    the quantized format (no F32 blowup in RAM).
  - F16->F32 via 64K lookup table (256 KB, built once)

  All byte/element offsets are Int64-capable (large models). }

{$POINTERMATH ON}

interface

uses
  System.SysUtils, System.Math, System.Threading;

const
  QK = 32;    // block size of the legacy GGML quantizations (Q4_0/Q4_1/Q8_0)
  QK_K = 256; // super-block size of the K-quantizations (Q4_K/Q5_K/Q6_K)

type
  { GGML tensor types (subset), values = GGUF/GGML enum }
  TGgmlType = (gtF32 = 0, gtF16 = 1, gtQ4_0 = 2, gtQ4_1 = 3, gtQ8_0 = 8,
    gtQ4_K = 12, gtQ5_K = 13, gtQ6_K = 14);

  { Quantized (or F32/F16) 2D tensor, stored row by row.
    Rows/Cols individually < 2^31, total size Int64. }
  TQTensor = record
    Typ: TGgmlType;
    Rows, Cols: Integer;
    Data: TBytes;
    class function RowBytesOf(ATyp: TGgmlType; ACols: Integer): Int64; static;
    function RowBytes: Int64;
    function TotalBytes: Int64;
    function IsEmpty: Boolean;
    { y[0..Rows) = W * x[0..Cols) - fused kernel, parallelized }
    procedure MatVec(Y, X: PSingle);
    { Dequantize one row to F32 (e.g. embedding lookup) }
    procedure DequantRow(Row: Integer; Dst: PSingle);
  end;

function HalfToFloat(H: Word): Single;

{ F32 base kernels }
function DotF32(A, B: PSingle; N: Integer): Single;
procedure MatVecF32(Y, W, X: PSingle; Rows, Cols: Integer; Bias: PSingle);
procedure AddVec(A, B: PSingle; N: Integer);
procedure MulVec(A, B: PSingle; N: Integer);
procedure ScaleVec(A: PSingle; S: Single; N: Integer);
procedure CopyVec(Dst, Src: PSingle; N: Integer);
procedure FillVec(A: PSingle; V: Single; N: Integer);
procedure SoftmaxVec(X: PSingle; N: Integer);
procedure RmsNormVec(Y, X, W: PSingle; N: Integer; Eps: Single);
procedure LayerNormVec(Y, X, W, B: PSingle; N: Integer; Eps: Single);
procedure SiluVec(X: PSingle; N: Integer);
procedure GeluVec(X: PSingle; N: Integer);
function ArgMax(X: PSingle; N: Integer): Integer;

{ Parallelism threshold: below this, thread fan-out does not pay off }
const
  PAR_MIN_WORK = 64 * 1024;

implementation

var
  GHalf: array [0 .. 65535] of Single;

function HalfToFloat(H: Word): Single;
begin
  Result := GHalf[H];
end;

function HalfToFloatCompute(H: Word): Single;
var
  Sign, Exp, Mant, Bits: UInt32;
begin
  Sign := UInt32(H and $8000) shl 16;
  Exp := (H shr 10) and $1F;
  Mant := H and $3FF;
  if Exp = 31 then
    Bits := Sign or $7F800000 or (Mant shl 13)
  else if Exp = 0 then
  begin
    if Mant = 0 then
      Bits := Sign
    else
    begin
      Exp := 127 - 15 + 1;
      while (Mant and $400) = 0 do
      begin
        Mant := Mant shl 1;
        Dec(Exp);
      end;
      Mant := Mant and $3FF;
      Bits := Sign or (Exp shl 23) or (Mant shl 13);
    end;
  end
  else
    Bits := Sign or (((Exp + 127) - 15) shl 23) or (Mant shl 13);
  Move(Bits, Result, 4);
end;

procedure InitHalfTable;
var
  I: Integer;
begin
  for I := 0 to 65535 do
    GHalf[I] := HalfToFloatCompute(Word(I));
end;

{ ---------- F32 kernels ---------- }

function DotF32(A, B: PSingle; N: Integer): Single;
var
  I, N4: Integer;
  S0, S1, S2, S3: Double;
begin
  S0 := 0; S1 := 0; S2 := 0; S3 := 0;
  N4 := N and not 3;
  I := 0;
  while I < N4 do
  begin
    S0 := S0 + A[I] * B[I];
    S1 := S1 + A[I + 1] * B[I + 1];
    S2 := S2 + A[I + 2] * B[I + 2];
    S3 := S3 + A[I + 3] * B[I + 3];
    Inc(I, 4);
  end;
  while I < N do
  begin
    S0 := S0 + A[I] * B[I];
    Inc(I);
  end;
  Result := S0 + S1 + S2 + S3;
end;

procedure MatVecF32(Y, W, X: PSingle; Rows, Cols: Integer; Bias: PSingle);
var
  R: Integer;
begin
  if Int64(Rows) * Cols >= PAR_MIN_WORK then
    TParallel.&For(0, Rows - 1,
      procedure(RI: Integer)
      begin
        Y[RI] := DotF32(W + Int64(RI) * Cols, X, Cols);
        if Bias <> nil then
          Y[RI] := Y[RI] + Bias[RI];
      end)
  else
    for R := 0 to Rows - 1 do
    begin
      Y[R] := DotF32(W + Int64(R) * Cols, X, Cols);
      if Bias <> nil then
        Y[R] := Y[R] + Bias[R];
    end;
end;

procedure AddVec(A, B: PSingle; N: Integer);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    A[I] := A[I] + B[I];
end;

procedure MulVec(A, B: PSingle; N: Integer);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    A[I] := A[I] * B[I];
end;

procedure ScaleVec(A: PSingle; S: Single; N: Integer);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    A[I] := A[I] * S;
end;

procedure CopyVec(Dst, Src: PSingle; N: Integer);
begin
  if N > 0 then
    Move(Src^, Dst^, N * SizeOf(Single));
end;

procedure FillVec(A: PSingle; V: Single; N: Integer);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    A[I] := V;
end;

procedure SoftmaxVec(X: PSingle; N: Integer);
var
  I: Integer;
  M: Single;
  S: Double;
begin
  if N <= 0 then
    Exit;
  M := X[0];
  for I := 1 to N - 1 do
    if X[I] > M then
      M := X[I];
  S := 0;
  for I := 0 to N - 1 do
  begin
    X[I] := Exp(X[I] - M);
    S := S + X[I];
  end;
  if S > 0 then
    for I := 0 to N - 1 do
      X[I] := X[I] / S;
end;

procedure RmsNormVec(Y, X, W: PSingle; N: Integer; Eps: Single);
var
  I: Integer;
  S: Double;
  R: Single;
begin
  S := 0;
  for I := 0 to N - 1 do
    S := S + Double(X[I]) * X[I];
  R := 1.0 / Sqrt(S / N + Eps);
  for I := 0 to N - 1 do
    Y[I] := X[I] * R * W[I];
end;

procedure LayerNormVec(Y, X, W, B: PSingle; N: Integer; Eps: Single);
var
  I: Integer;
  M, V: Double;
  D, R: Single;
begin
  M := 0;
  for I := 0 to N - 1 do
    M := M + X[I];
  M := M / N;
  V := 0;
  for I := 0 to N - 1 do
  begin
    D := X[I] - M;
    V := V + Double(D) * D;
  end;
  R := 1.0 / Sqrt(V / N + Eps);
  for I := 0 to N - 1 do
    Y[I] := (X[I] - M) * R * W[I] + B[I];
end;

procedure SiluVec(X: PSingle; N: Integer);
var
  I: Integer;
  V: Single;
begin
  for I := 0 to N - 1 do
  begin
    V := X[I];
    X[I] := V / (1.0 + Exp(-V));
  end;
end;

procedure GeluVec(X: PSingle; N: Integer);
const
  S = 0.7978845608028654; // sqrt(2/pi)
var
  I: Integer;
  V: Single;
begin
  for I := 0 to N - 1 do
  begin
    V := X[I];
    X[I] := 0.5 * V * (1.0 + Tanh(S * (V + 0.044715 * V * V * V)));
  end;
end;

function ArgMax(X: PSingle; N: Integer): Integer;
var
  I: Integer;
begin
  Result := 0;
  for I := 1 to N - 1 do
    if X[I] > X[Result] then
      Result := I;
end;

{ ---------- Quantization ---------- }

{ Quantize activation x to int8: one scale factor per 32-element block.
  XQ: N int8 values, XS: N/32 scales, XSum: N/32 block sums of x (for Q4). }
procedure QuantizeActivation(X: PSingle; N: Integer; XQ: PShortInt;
  XS, XSum: PSingle);
var
  B, I, NB: Integer;
  AMax, V, D, ID: Single;
  S: Double;
begin
  NB := N div QK;
  for B := 0 to NB - 1 do
  begin
    AMax := 0;
    S := 0;
    for I := 0 to QK - 1 do
    begin
      V := X[B * QK + I];
      S := S + V;
      if Abs(V) > AMax then
        AMax := Abs(V);
    end;
    XSum[B] := S;
    D := AMax / 127.0;
    XS[B] := D;
    if D > 0 then
      ID := 1.0 / D
    else
      ID := 0;
    for I := 0 to QK - 1 do
      XQ[B * QK + I] := ShortInt(Round(X[B * QK + I] * ID));
  end;
end;

{ Q8_0 block: 2-byte F16 scale + 32 int8  (34 bytes / 32 values) }
function DotRowQ8_0(Row: PByte; XQ: PShortInt; XS: PSingle; NB: Integer): Single;
var
  B, I, SumI: Integer;
  P: PByte;
  Q: PShortInt;
  D: Single;
  Acc: Double;
begin
  Acc := 0;
  P := Row;
  for B := 0 to NB - 1 do
  begin
    D := GHalf[PWord(P)^];
    Q := PShortInt(P + 2);
    SumI := 0;
    for I := 0 to QK - 1 do
      SumI := SumI + Integer(Q[I]) * Integer(XQ[B * QK + I]);
    Acc := Acc + Double(D) * XS[B] * SumI;
    Inc(P, 34);
  end;
  Result := Acc;
end;

{ Q4_0 block: 2-byte F16 scale + 16 bytes of nibbles (18 bytes / 32 values)
  value i (0..15) = low nibble - 8, value i+16 = high nibble - 8 }
function DotRowQ4_0(Row: PByte; XQ: PShortInt; XS: PSingle; NB: Integer): Single;
var
  B, I, SumI, SumX: Integer;
  P, Q: PByte;
  D: Single;
  Acc: Double;
  X0: PShortInt;
begin
  Acc := 0;
  P := Row;
  for B := 0 to NB - 1 do
  begin
    D := GHalf[PWord(P)^];
    Q := P + 2;
    X0 := XQ + B * QK;
    SumI := 0;
    SumX := 0;
    for I := 0 to (QK div 2) - 1 do
    begin
      SumI := SumI + Integer(Q[I] and $0F) * Integer(X0[I]) +
        Integer(Q[I] shr 4) * Integer(X0[I + QK div 2]);
      SumX := SumX + Integer(X0[I]) + Integer(X0[I + QK div 2]);
    end;
    { (q-8)*x = q*x - 8*sum(x) }
    Acc := Acc + Double(D) * XS[B] * (SumI - 8 * SumX);
    Inc(P, 18);
  end;
  Result := Acc;
end;

{ Q4_1 block: F16 d + F16 m + 16 bytes of nibbles (20 bytes / 32 values)
  value = q*d + m  ->  dot += d*sum(q*x) + m*sum(x)  (sum(x) in F32) }
function DotRowQ4_1(Row: PByte; XQ: PShortInt; XS, XSum: PSingle;
  NB: Integer): Single;
var
  B, I, SumI: Integer;
  P, Q: PByte;
  D, M: Single;
  Acc: Double;
  X0: PShortInt;
begin
  Acc := 0;
  P := Row;
  for B := 0 to NB - 1 do
  begin
    D := GHalf[PWord(P)^];
    M := GHalf[PWord(P + 2)^];
    Q := P + 4;
    X0 := XQ + B * QK;
    SumI := 0;
    for I := 0 to (QK div 2) - 1 do
      SumI := SumI + Integer(Q[I] and $0F) * Integer(X0[I]) +
        Integer(Q[I] shr 4) * Integer(X0[I + QK div 2]);
    Acc := Acc + Double(D) * XS[B] * SumI + Double(M) * XSum[B];
    Inc(P, 20);
  end;
  Result := Acc;
end;

function DotRowF16(Row: PWord; X: PSingle; N: Integer): Single;
var
  I: Integer;
  Acc: Double;
begin
  Acc := 0;
  for I := 0 to N - 1 do
    Acc := Acc + Double(GHalf[Row[I]]) * X[I];
  Result := Acc;
end;

{ ---------- K-quantizations (super-block = 256 values) ----------
  Way A: dequantize each super-block to F32 on the fly, then reuse DotF32.
  Bit layouts follow llama.cpp/ggml-quants.c (new QK_K = 256 format). }

{ Q4_K/Q5_K sub-block scale+min: 6-bit values packed into 12 bytes }
procedure GetScaleMinK4(J: Integer; Q: PByte; out D, M: Byte);
begin
  if J < 4 then
  begin
    D := Q[J] and 63;
    M := Q[J + 4] and 63;
  end
  else
  begin
    D := (Q[J + 4] and $0F) or ((Q[J - 4] shr 6) shl 4);
    M := (Q[J + 4] shr 4) or ((Q[J] shr 6) shl 4);
  end;
end;

{ Q4_K super-block: d,dmin (F16) + 12 scale bytes + 128 quant bytes = 144 B }
procedure DequantSuperQ4_K(P: PByte; Dst: PSingle);
var
  D, Dmin, D1, D2, M1, M2: Single;
  Scales, Q: PByte;
  J, L, Is_, YBase: Integer;
  Sc, M: Byte;
begin
  D := GHalf[PWord(P)^];
  Dmin := GHalf[PWord(P + 2)^];
  Scales := P + 4;
  Q := P + 16;
  Is_ := 0;
  YBase := 0;
  for J := 0 to QK_K div 64 - 1 do
  begin
    GetScaleMinK4(Is_, Scales, Sc, M);
    D1 := D * Sc; M1 := Dmin * M;
    GetScaleMinK4(Is_ + 1, Scales, Sc, M);
    D2 := D * Sc; M2 := Dmin * M;
    for L := 0 to 31 do
      Dst[YBase + L] := D1 * (Q[L] and $0F) - M1;
    for L := 0 to 31 do
      Dst[YBase + 32 + L] := D2 * (Q[L] shr 4) - M2;
    Inc(Q, 32);
    Inc(Is_, 2);
    Inc(YBase, 64);
  end;
end;

{ Q5_K super-block: d,dmin (F16) + 12 scale + 32 high-bit + 128 quant = 176 B }
procedure DequantSuperQ5_K(P: PByte; Dst: PSingle);
var
  D, Dmin, D1, D2, M1, M2: Single;
  Scales, Qh, Ql: PByte;
  J, L, Is_, YBase, U1, U2: Integer;
  Sc, M: Byte;
begin
  D := GHalf[PWord(P)^];
  Dmin := GHalf[PWord(P + 2)^];
  Scales := P + 4;
  Qh := P + 16;
  Ql := P + 48;
  Is_ := 0;
  U1 := 1;
  U2 := 2;
  YBase := 0;
  for J := 0 to QK_K div 64 - 1 do
  begin
    GetScaleMinK4(Is_, Scales, Sc, M);
    D1 := D * Sc; M1 := Dmin * M;
    GetScaleMinK4(Is_ + 1, Scales, Sc, M);
    D2 := D * Sc; M2 := Dmin * M;
    for L := 0 to 31 do
      if (Qh[L] and U1) <> 0 then
        Dst[YBase + L] := D1 * ((Ql[L] and $0F) + 16) - M1
      else
        Dst[YBase + L] := D1 * (Ql[L] and $0F) - M1;
    for L := 0 to 31 do
      if (Qh[L] and U2) <> 0 then
        Dst[YBase + 32 + L] := D2 * ((Ql[L] shr 4) + 16) - M2
      else
        Dst[YBase + 32 + L] := D2 * (Ql[L] shr 4) - M2;
    Inc(Ql, 32);
    Inc(Is_, 2);
    U1 := U1 shl 2;
    U2 := U2 shl 2;
    Inc(YBase, 64);
  end;
end;

{ Q6_K super-block: 128 low + 64 high + 16 int8 scales + d (F16) = 210 B }
procedure DequantSuperQ6_K(P: PByte; Dst: PSingle);
var
  D: Single;
  Ql, Qh: PByte;
  Sc: PShortInt;
  N, L, Is_, YBase: Integer;
begin
  D := GHalf[PWord(P + 208)^];
  Ql := P;
  Qh := P + 128;
  Sc := PShortInt(P + 192);
  YBase := 0;
  for N := 0 to QK_K div 128 - 1 do
  begin
    for L := 0 to 31 do
    begin
      Is_ := L div 16;
      Dst[YBase + L] := D * Sc[Is_] *
        (Integer((Ql[L] and $0F) or (((Qh[L] shr 0) and 3) shl 4)) - 32);
      Dst[YBase + L + 32] := D * Sc[Is_ + 2] *
        (Integer((Ql[L + 32] and $0F) or (((Qh[L] shr 2) and 3) shl 4)) - 32);
      Dst[YBase + L + 64] := D * Sc[Is_ + 4] *
        (Integer((Ql[L] shr 4) or (((Qh[L] shr 4) and 3) shl 4)) - 32);
      Dst[YBase + L + 96] := D * Sc[Is_ + 6] *
        (Integer((Ql[L + 32] shr 4) or (((Qh[L] shr 6) and 3) shl 4)) - 32);
    end;
    Inc(Ql, 64);
    Inc(Qh, 32);
    Inc(Sc, 8);
    Inc(YBase, 128);
  end;
end;

function DotRowQ4_K(Row: PByte; X: PSingle; NCols: Integer): Single;
var
  SB, NSB: Integer;
  P: PByte;
  Acc: Double;
  Blk: array [0 .. QK_K - 1] of Single;
begin
  Acc := 0;
  NSB := NCols div QK_K;
  P := Row;
  for SB := 0 to NSB - 1 do
  begin
    DequantSuperQ4_K(P, @Blk[0]);
    Acc := Acc + DotF32(@Blk[0], X + SB * QK_K, QK_K);
    Inc(P, 144);
  end;
  Result := Acc;
end;

function DotRowQ5_K(Row: PByte; X: PSingle; NCols: Integer): Single;
var
  SB, NSB: Integer;
  P: PByte;
  Acc: Double;
  Blk: array [0 .. QK_K - 1] of Single;
begin
  Acc := 0;
  NSB := NCols div QK_K;
  P := Row;
  for SB := 0 to NSB - 1 do
  begin
    DequantSuperQ5_K(P, @Blk[0]);
    Acc := Acc + DotF32(@Blk[0], X + SB * QK_K, QK_K);
    Inc(P, 176);
  end;
  Result := Acc;
end;

function DotRowQ6_K(Row: PByte; X: PSingle; NCols: Integer): Single;
var
  SB, NSB: Integer;
  P: PByte;
  Acc: Double;
  Blk: array [0 .. QK_K - 1] of Single;
begin
  Acc := 0;
  NSB := NCols div QK_K;
  P := Row;
  for SB := 0 to NSB - 1 do
  begin
    DequantSuperQ6_K(P, @Blk[0]);
    Acc := Acc + DotF32(@Blk[0], X + SB * QK_K, QK_K);
    Inc(P, 210);
  end;
  Result := Acc;
end;

{ TQTensor }

class function TQTensor.RowBytesOf(ATyp: TGgmlType; ACols: Integer): Int64;
begin
  case ATyp of
    gtF32:  Result := Int64(ACols) * 4;
    gtF16:  Result := Int64(ACols) * 2;
    gtQ4_0: Result := Int64(ACols div QK) * 18;
    gtQ4_1: Result := Int64(ACols div QK) * 20;
    gtQ8_0: Result := Int64(ACols div QK) * 34;
    gtQ4_K: Result := Int64(ACols div QK_K) * 144;
    gtQ5_K: Result := Int64(ACols div QK_K) * 176;
    gtQ6_K: Result := Int64(ACols div QK_K) * 210;
  else
    raise Exception.Create('TQTensor: unsupported GGML type');
  end;
end;

function TQTensor.RowBytes: Int64;
begin
  Result := RowBytesOf(Typ, Cols);
end;

function TQTensor.TotalBytes: Int64;
begin
  Result := RowBytes * Rows;
end;

function TQTensor.IsEmpty: Boolean;
begin
  Result := Length(Data) = 0;
end;

procedure TQTensor.MatVec(Y, X: PSingle);
var
  NB, R: Integer;
  XQ: TArray<ShortInt>;
  XS, XSum: TArray<Single>;
  RB: Int64;
  Base: PByte;
  LTyp: TGgmlType;
  PXQ: PShortInt;
  PXS, PXSum: PSingle;
  LCols: Integer;
begin
  LTyp := Typ;
  LCols := Cols;
  RB := RowBytes;
  Base := PByte(Data);
  NB := LCols div QK;
  PXQ := nil; PXS := nil; PXSum := nil;
  if LTyp in [gtQ4_0, gtQ4_1, gtQ8_0] then
  begin
    SetLength(XQ, LCols);
    SetLength(XS, NB);
    SetLength(XSum, NB);
    QuantizeActivation(X, LCols, @XQ[0], @XS[0], @XSum[0]);
    PXQ := @XQ[0];
    PXS := @XS[0];
    PXSum := @XSum[0];
  end;
  if Int64(Rows) * LCols >= PAR_MIN_WORK then
    TParallel.&For(0, Rows - 1,
      procedure(RI: Integer)
      var
        P: PByte;
      begin
        P := Base + Int64(RI) * RB;
        case LTyp of
          gtF32:  Y[RI] := DotF32(PSingle(P), X, LCols);
          gtF16:  Y[RI] := DotRowF16(PWord(P), X, LCols);
          gtQ8_0: Y[RI] := DotRowQ8_0(P, PXQ, PXS, NB);
          gtQ4_0: Y[RI] := DotRowQ4_0(P, PXQ, PXS, NB);
          gtQ4_1: Y[RI] := DotRowQ4_1(P, PXQ, PXS, PXSum, NB);
          gtQ4_K: Y[RI] := DotRowQ4_K(P, X, LCols);
          gtQ5_K: Y[RI] := DotRowQ5_K(P, X, LCols);
          gtQ6_K: Y[RI] := DotRowQ6_K(P, X, LCols);
        end;
      end)
  else
    for R := 0 to Rows - 1 do
    begin
      case LTyp of
        gtF32:  Y[R] := DotF32(PSingle(Base + Int64(R) * RB), X, LCols);
        gtF16:  Y[R] := DotRowF16(PWord(Base + Int64(R) * RB), X, LCols);
        gtQ8_0: Y[R] := DotRowQ8_0(Base + Int64(R) * RB, PXQ, PXS, NB);
        gtQ4_0: Y[R] := DotRowQ4_0(Base + Int64(R) * RB, PXQ, PXS, NB);
        gtQ4_1: Y[R] := DotRowQ4_1(Base + Int64(R) * RB, PXQ, PXS, PXSum, NB);
        gtQ4_K: Y[R] := DotRowQ4_K(Base + Int64(R) * RB, X, LCols);
        gtQ5_K: Y[R] := DotRowQ5_K(Base + Int64(R) * RB, X, LCols);
        gtQ6_K: Y[R] := DotRowQ6_K(Base + Int64(R) * RB, X, LCols);
      end;
    end;
end;

procedure TQTensor.DequantRow(Row: Integer; Dst: PSingle);
var
  P, Q: PByte;
  I, B, NB: Integer;
  D, M: Single;
begin
  P := PByte(Data) + Int64(Row) * RowBytes;
  case Typ of
    gtF32:
      Move(P^, Dst^, Cols * SizeOf(Single));
    gtF16:
      for I := 0 to Cols - 1 do
        Dst[I] := GHalf[PWord(P)[I]];
    gtQ8_0:
      begin
        NB := Cols div QK;
        for B := 0 to NB - 1 do
        begin
          D := GHalf[PWord(P)^];
          Q := P + 2;
          for I := 0 to QK - 1 do
            Dst[B * QK + I] := Integer(PShortInt(Q)[I]) * D;
          Inc(P, 34);
        end;
      end;
    gtQ4_0:
      begin
        NB := Cols div QK;
        for B := 0 to NB - 1 do
        begin
          D := GHalf[PWord(P)^];
          Q := P + 2;
          for I := 0 to (QK div 2) - 1 do
          begin
            Dst[B * QK + I] := (Integer(Q[I] and $0F) - 8) * D;
            Dst[B * QK + I + QK div 2] := (Integer(Q[I] shr 4) - 8) * D;
          end;
          Inc(P, 18);
        end;
      end;
    gtQ4_1:
      begin
        NB := Cols div QK;
        for B := 0 to NB - 1 do
        begin
          D := GHalf[PWord(P)^];
          M := GHalf[PWord(P + 2)^];
          Q := P + 4;
          for I := 0 to (QK div 2) - 1 do
          begin
            Dst[B * QK + I] := Integer(Q[I] and $0F) * D + M;
            Dst[B * QK + I + QK div 2] := Integer(Q[I] shr 4) * D + M;
          end;
          Inc(P, 20);
        end;
      end;
    gtQ4_K:
      begin
        NB := Cols div QK_K;
        for B := 0 to NB - 1 do
        begin
          DequantSuperQ4_K(P, Dst + B * QK_K);
          Inc(P, 144);
        end;
      end;
    gtQ5_K:
      begin
        NB := Cols div QK_K;
        for B := 0 to NB - 1 do
        begin
          DequantSuperQ5_K(P, Dst + B * QK_K);
          Inc(P, 176);
        end;
      end;
    gtQ6_K:
      begin
        NB := Cols div QK_K;
        for B := 0 to NB - 1 do
        begin
          DequantSuperQ6_K(P, Dst + B * QK_K);
          Inc(P, 210);
        end;
      end;
  end;
end;

initialization
  InitHalfTable;

end.
