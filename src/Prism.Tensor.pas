unit Prism.Tensor;

{ Batch kernels for TRAINING (forward and backward passes).
  Math port in the style of llm.c (GPT-2), pointer-based.

  Conventions:
  - All tensors are flat Single buffers, the caller passes pointers.
  - B = batch, T = sequence length, C = channels, NH = heads, OC = out channels
  - Backward functions ACCUMULATE into the d* buffers (zero them first!). }

{$POINTERMATH ON}

interface

uses
  System.Math, System.Threading, Prism.Vector;

const
  LN_EPS = 1e-5;

procedure EncoderForward(OutP: PSingle; Inp: PInteger; Wte, Wpe: PSingle;
  B, T, C: Integer);
procedure EncoderBackward(DWte, DWpe, DOut: PSingle; Inp: PInteger;
  B, T, C: Integer);

procedure LayerNormForward(OutP, Mean, Rstd, Inp, W, Bias: PSingle;
  B, T, C: Integer);
procedure LayerNormBackward(DInp, DW, DB, DOut, Inp, W, Mean, Rstd: PSingle;
  B, T, C: Integer);

procedure MatMulForward(OutP, Inp, W, Bias: PSingle; B, T, C, OC: Integer);
procedure MatMulBackward(DInp, DW, DBias, DOut, Inp, W: PSingle;
  B, T, C, OC: Integer);

procedure AttentionForward(OutP, Preatt, Att, Inp: PSingle;
  B, T, C, NH: Integer);
procedure AttentionBackward(DInp, DPreatt, DAtt, DOut, Inp, Att: PSingle;
  B, T, C, NH: Integer);

procedure GeluForward(OutP, Inp: PSingle; N: Int64);
procedure GeluBackward(DInp, Inp, DOut: PSingle; N: Int64);

procedure ResidualForward(OutP, A, B_: PSingle; N: Int64);
procedure ResidualBackward(DA, DB, DOut: PSingle; N: Int64);

{ residual3 = residual2 + gate * expertOut  (MoE, gate per token) }
procedure ScaledResidualForward(OutP, A, B_: PSingle; Gate: Single; N: Integer);

procedure SoftmaxForward(Probs, Logits: PSingle; B, T, V: Integer);
procedure CrossEntropyForward(Losses, Probs: PSingle; Targets: PInteger;
  B, T, V: Integer);
procedure CrossEntropySoftmaxBackward(DLogits, DLosses, Probs: PSingle;
  Targets: PInteger; B, T, V: Integer);

implementation

const
  GELU_S = 0.7978845608028654; // sqrt(2/pi)

procedure EncoderForward(OutP: PSingle; Inp: PInteger; Wte, Wpe: PSingle;
  B, T, C: Integer);
var
  BI, TI, I: Integer;
  O, WteRow, WpeRow: PSingle;
begin
  for BI := 0 to B - 1 do
    for TI := 0 to T - 1 do
    begin
      O := OutP + (Int64(BI) * T + TI) * C;
      WteRow := Wte + Int64(Inp[BI * T + TI]) * C;
      WpeRow := Wpe + Int64(TI) * C;
      for I := 0 to C - 1 do
        O[I] := WteRow[I] + WpeRow[I];
    end;
end;

procedure EncoderBackward(DWte, DWpe, DOut: PSingle; Inp: PInteger;
  B, T, C: Integer);
var
  BI, TI, I: Integer;
  D, DTe, DPe: PSingle;
begin
  for BI := 0 to B - 1 do
    for TI := 0 to T - 1 do
    begin
      D := DOut + (Int64(BI) * T + TI) * C;
      DTe := DWte + Int64(Inp[BI * T + TI]) * C;
      DPe := DWpe + Int64(TI) * C;
      for I := 0 to C - 1 do
      begin
        DTe[I] := DTe[I] + D[I];
        DPe[I] := DPe[I] + D[I];
      end;
    end;
end;

procedure LayerNormForward(OutP, Mean, Rstd, Inp, W, Bias: PSingle;
  B, T, C: Integer);
var
  BT, I: Integer;
  X, O: PSingle;
  M, V: Double;
  D, R: Single;
begin
  for BT := 0 to B * T - 1 do
  begin
    X := Inp + Int64(BT) * C;
    O := OutP + Int64(BT) * C;
    M := 0;
    for I := 0 to C - 1 do
      M := M + X[I];
    M := M / C;
    V := 0;
    for I := 0 to C - 1 do
    begin
      D := X[I] - M;
      V := V + Double(D) * D;
    end;
    R := 1.0 / Sqrt(V / C + LN_EPS);
    for I := 0 to C - 1 do
      O[I] := (X[I] - M) * R * W[I] + Bias[I];
    Mean[BT] := M;
    Rstd[BT] := R;
  end;
end;

procedure LayerNormBackward(DInp, DW, DB, DOut, Inp, W, Mean, Rstd: PSingle;
  B, T, C: Integer);
var
  BT, I: Integer;
  X, DO_, DX: PSingle;
  M, R, NormI, DnormI: Single;
  DnormMean, DnormNormMean: Double;
begin
  for BT := 0 to B * T - 1 do
  begin
    X := Inp + Int64(BT) * C;
    DO_ := DOut + Int64(BT) * C;
    DX := DInp + Int64(BT) * C;
    M := Mean[BT];
    R := Rstd[BT];
    DnormMean := 0;
    DnormNormMean := 0;
    for I := 0 to C - 1 do
    begin
      NormI := (X[I] - M) * R;
      DnormI := W[I] * DO_[I];
      DnormMean := DnormMean + DnormI;
      DnormNormMean := DnormNormMean + DnormI * NormI;
    end;
    DnormMean := DnormMean / C;
    DnormNormMean := DnormNormMean / C;
    for I := 0 to C - 1 do
    begin
      NormI := (X[I] - M) * R;
      DnormI := W[I] * DO_[I];
      DW[I] := DW[I] + NormI * DO_[I];
      DB[I] := DB[I] + DO_[I];
      DX[I] := DX[I] + (DnormI - DnormMean - NormI * DnormNormMean) * R;
    end;
  end;
end;

procedure MatMulForward(OutP, Inp, W, Bias: PSingle; B, T, C, OC: Integer);
begin
  TParallel.&For(0, B * T - 1,
    procedure(BT: Integer)
    var
      O, X: PSingle;
      OI: Integer;
    begin
      X := Inp + Int64(BT) * C;
      O := OutP + Int64(BT) * OC;
      for OI := 0 to OC - 1 do
      begin
        O[OI] := DotF32(W + Int64(OI) * C, X, C);
        if Bias <> nil then
          O[OI] := O[OI] + Bias[OI];
      end;
    end);
end;

procedure MatMulBackward(DInp, DW, DBias, DOut, Inp, W: PSingle;
  B, T, C, OC: Integer);
begin
  { Phase 1: dinp += dout * W  (parallel over time steps) }
  if DInp <> nil then
    TParallel.&For(0, B * T - 1,
      procedure(BT: Integer)
      var
        DX, DO_, WRow: PSingle;
        OI, I: Integer;
        D: Single;
      begin
        DX := DInp + Int64(BT) * C;
        DO_ := DOut + Int64(BT) * OC;
        for OI := 0 to OC - 1 do
        begin
          D := DO_[OI];
          if D <> 0 then
          begin
            WRow := W + Int64(OI) * C;
            for I := 0 to C - 1 do
              DX[I] := DX[I] + WRow[I] * D;
          end;
        end;
      end);
  { Phase 2: dW += dout^T * inp, dbias += sum(dout)  (parallel over OC) }
  TParallel.&For(0, OC - 1,
    procedure(OI: Integer)
    var
      DWRow, X: PSingle;
      BT, I: Integer;
      D: Single;
      BSum: Double;
    begin
      DWRow := DW + Int64(OI) * C;
      BSum := 0;
      for BT := 0 to B * T - 1 do
      begin
        D := DOut[Int64(BT) * OC + OI];
        if D <> 0 then
        begin
          BSum := BSum + D;
          X := Inp + Int64(BT) * C;
          for I := 0 to C - 1 do
            DWRow[I] := DWRow[I] + X[I] * D;
        end;
      end;
      if DBias <> nil then
        DBias[OI] := DBias[OI] + BSum;
    end);
end;

procedure AttentionForward(OutP, Preatt, Att, Inp: PSingle;
  B, T, C, NH: Integer);
var
  HS: Integer;
  Scale: Single;
begin
  HS := C div NH;
  Scale := 1.0 / Sqrt(HS);
  { Inp = QKV [B,T,3C]; parallel over (b, h) }
  TParallel.&For(0, B * NH - 1,
    procedure(BH: Integer)
    var
      BI, H, TI, T2, I: Integer;
      Q, K, V, O, PA, A: PSingle;
      S, MaxV: Single;
      Sum: Double;
    begin
      BI := BH div NH;
      H := BH mod NH;
      for TI := 0 to T - 1 do
      begin
        Q := Inp + (Int64(BI) * T + TI) * 3 * C + H * HS;
        PA := Preatt + ((Int64(BI) * NH + H) * T + TI) * T;
        A := Att + ((Int64(BI) * NH + H) * T + TI) * T;
        MaxV := -1e30;
        for T2 := 0 to TI do
        begin
          K := Inp + (Int64(BI) * T + T2) * 3 * C + C + H * HS;
          S := 0;
          for I := 0 to HS - 1 do
            S := S + Q[I] * K[I];
          S := S * Scale;
          PA[T2] := S;
          if S > MaxV then
            MaxV := S;
        end;
        Sum := 0;
        for T2 := 0 to TI do
        begin
          A[T2] := Exp(PA[T2] - MaxV);
          Sum := Sum + A[T2];
        end;
        if Sum > 0 then
          for T2 := 0 to TI do
            A[T2] := A[T2] / Sum;
        for T2 := TI + 1 to T - 1 do
          A[T2] := 0;
        O := OutP + (Int64(BI) * T + TI) * C + H * HS;
        for I := 0 to HS - 1 do
          O[I] := 0;
        for T2 := 0 to TI do
        begin
          V := Inp + (Int64(BI) * T + T2) * 3 * C + 2 * C + H * HS;
          S := A[T2];
          for I := 0 to HS - 1 do
            O[I] := O[I] + S * V[I];
        end;
      end;
    end);
end;

procedure AttentionBackward(DInp, DPreatt, DAtt, DOut, Inp, Att: PSingle;
  B, T, C, NH: Integer);
var
  HS: Integer;
  Scale: Single;
begin
  HS := C div NH;
  Scale := 1.0 / Sqrt(HS);
  { parallel over b (within one b the head slices are disjoint,
    but dK/dV writes overlap across t -> serial per b) }
  TParallel.&For(0, B - 1,
    procedure(BI: Integer)
    var
      H, TI, T2, T3, I: Integer;
      Q, K, V, DQ, DK, DV, DO_, A, DA, DPA: PSingle;
      D, LocalD: Single;
    begin
      for H := 0 to NH - 1 do
        for TI := 0 to T - 1 do
        begin
          A := Att + ((Int64(BI) * NH + H) * T + TI) * T;
          DA := DAtt + ((Int64(BI) * NH + H) * T + TI) * T;
          DPA := DPreatt + ((Int64(BI) * NH + H) * T + TI) * T;
          DO_ := DOut + (Int64(BI) * T + TI) * C + H * HS;
          Q := Inp + (Int64(BI) * T + TI) * 3 * C + H * HS;
          DQ := DInp + (Int64(BI) * T + TI) * 3 * C + H * HS;
          { dV and dAtt }
          for T2 := 0 to TI do
          begin
            V := Inp + (Int64(BI) * T + T2) * 3 * C + 2 * C + H * HS;
            DV := DInp + (Int64(BI) * T + T2) * 3 * C + 2 * C + H * HS;
            D := 0;
            for I := 0 to HS - 1 do
            begin
              D := D + V[I] * DO_[I];
              DV[I] := DV[I] + A[T2] * DO_[I];
            end;
            DA[T2] := DA[T2] + D;
          end;
          { Softmax backward: dPreatt }
          for T2 := 0 to TI do
            for T3 := 0 to TI do
            begin
              if T2 = T3 then
                LocalD := A[T2] * (1.0 - A[T3])
              else
                LocalD := -A[T2] * A[T3];
              DPA[T3] := DPA[T3] + LocalD * DA[T2];
            end;
          { dQ and dK }
          for T2 := 0 to TI do
          begin
            K := Inp + (Int64(BI) * T + T2) * 3 * C + C + H * HS;
            DK := DInp + (Int64(BI) * T + T2) * 3 * C + C + H * HS;
            D := DPA[T2] * Scale;
            for I := 0 to HS - 1 do
            begin
              DQ[I] := DQ[I] + K[I] * D;
              DK[I] := DK[I] + Q[I] * D;
            end;
          end;
        end;
    end);
end;

procedure GeluForward(OutP, Inp: PSingle; N: Int64);
var
  I: Int64;
  V: Single;
begin
  for I := 0 to N - 1 do
  begin
    V := Inp[I];
    OutP[I] := 0.5 * V * (1.0 + Tanh(GELU_S * (V + 0.044715 * V * V * V)));
  end;
end;

procedure GeluBackward(DInp, Inp, DOut: PSingle; N: Int64);
var
  I: Int64;
  X, Cube, TanhArg, TanhOut, CoshOut, Sech2, LocalGrad: Single;
begin
  for I := 0 to N - 1 do
  begin
    X := Inp[I];
    Cube := 0.044715 * X * X * X;
    TanhArg := GELU_S * (X + Cube);
    TanhOut := Tanh(TanhArg);
    CoshOut := Cosh(TanhArg);
    Sech2 := 1.0 / (CoshOut * CoshOut);
    LocalGrad := 0.5 * (1.0 + TanhOut) + X * 0.5 * Sech2 * GELU_S *
      (1.0 + 3.0 * 0.044715 * X * X);
    DInp[I] := DInp[I] + LocalGrad * DOut[I];
  end;
end;

procedure ResidualForward(OutP, A, B_: PSingle; N: Int64);
var
  I: Int64;
begin
  for I := 0 to N - 1 do
    OutP[I] := A[I] + B_[I];
end;

procedure ResidualBackward(DA, DB, DOut: PSingle; N: Int64);
var
  I: Int64;
begin
  for I := 0 to N - 1 do
  begin
    DA[I] := DA[I] + DOut[I];
    DB[I] := DB[I] + DOut[I];
  end;
end;

procedure ScaledResidualForward(OutP, A, B_: PSingle; Gate: Single; N: Integer);
var
  I: Integer;
begin
  for I := 0 to N - 1 do
    OutP[I] := A[I] + Gate * B_[I];
end;

procedure SoftmaxForward(Probs, Logits: PSingle; B, T, V: Integer);
begin
  TParallel.&For(0, B * T - 1,
    procedure(BT: Integer)
    var
      P, L: PSingle;
      I: Integer;
      MaxV: Single;
      Sum: Double;
    begin
      L := Logits + Int64(BT) * V;
      P := Probs + Int64(BT) * V;
      MaxV := L[0];
      for I := 1 to V - 1 do
        if L[I] > MaxV then
          MaxV := L[I];
      Sum := 0;
      for I := 0 to V - 1 do
      begin
        P[I] := Exp(L[I] - MaxV);
        Sum := Sum + P[I];
      end;
      for I := 0 to V - 1 do
        P[I] := P[I] / Sum;
    end);
end;

procedure CrossEntropyForward(Losses, Probs: PSingle; Targets: PInteger;
  B, T, V: Integer);
var
  BT: Integer;
  P: Single;
begin
  for BT := 0 to B * T - 1 do
  begin
    P := Probs[Int64(BT) * V + Targets[BT]];
    if P < 1e-10 then
      P := 1e-10;
    Losses[BT] := -Ln(P);
  end;
end;

procedure CrossEntropySoftmaxBackward(DLogits, DLosses, Probs: PSingle;
  Targets: PInteger; B, T, V: Integer);
begin
  TParallel.&For(0, B * T - 1,
    procedure(BT: Integer)
    var
      DL, P: PSingle;
      I, Tgt: Integer;
      D: Single;
    begin
      DL := DLogits + Int64(BT) * V;
      P := Probs + Int64(BT) * V;
      D := DLosses[BT];
      Tgt := Targets[BT];
      for I := 0 to V - 1 do
        DL[I] := DL[I] + P[I] * D;
      DL[Tgt] := DL[Tgt] - D;
    end);
end;

end.
