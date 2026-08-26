program PrismBench;

{ Prism GPU verification and benchmark tool.

  Two jobs, both of which the GPU work needed and neither of which the server
  is the right place for:

  1. VERIFY. Runs the Vulkan backend's self-test, which compares every
     quantized kernel against the CPU reference in Prism.Vector. The GGUF bit
     layouts in the shader were transcribed by hand, so this is the check that
     says whether the GPU path is trustworthy at all.

  2. MEASURE. Times MatVec on CPU and GPU at realistic transformer shapes for
     each quantization, so a speed claim can be a number instead of a guess.

  Usage:
    PrismBench                     verify + benchmark on the best GPU
    PrismBench --device 1          pick a GPU (index or name substring)
    PrismBench --budget 2048       cap the VRAM budget, in MB
    PrismBench --iters 50          timed iterations per case
    PrismBench --verify-only       skip the benchmark }

{$APPTYPE CONSOLE}
{$POINTERMATH ON}   // BuildTensor indexes PByte/PWord/PSingle directly

uses
  System.SysUtils,
  System.Classes,
  System.Diagnostics,
  System.Math,
{$IFDEF MSWINDOWS}
  Winapi.Windows,
{$ELSE}
  Posix.Stdlib,
{$ENDIF}
  Prism.Types in '..\src\Prism.Types.pas',
  Prism.Vector in '..\src\Prism.Vector.pas',
  Prism.Vulkan.Api in '..\src\Prism.Vulkan.Api.pas',
  Prism.Vulkan in '..\src\Prism.Vulkan.pas',
  Prism.Gpu in '..\src\Prism.Gpu.pas';

type
  TCase = record
    Typ: TGgmlType;
    Name: string;
    Rows, Cols: Integer;
  end;

const
  { Shapes taken from real models: 4096 is Llama-7B's hidden size, 11008 its
    FFN width, 32000 its vocabulary. }
  CASES: array [0 .. 7] of TCase = (
    (Typ: gtQ4_K; Name: 'Q4_K'; Rows: 4096;  Cols: 4096),
    (Typ: gtQ4_K; Name: 'Q4_K'; Rows: 11008; Cols: 4096),
    (Typ: gtQ6_K; Name: 'Q6_K'; Rows: 4096;  Cols: 4096),
    (Typ: gtQ5_K; Name: 'Q5_K'; Rows: 4096;  Cols: 4096),
    (Typ: gtQ8_0; Name: 'Q8_0'; Rows: 4096;  Cols: 4096),
    (Typ: gtQ4_0; Name: 'Q4_0'; Rows: 4096;  Cols: 4096),
    (Typ: gtF16;  Name: 'F16';  Rows: 4096;  Cols: 4096),
    (Typ: gtQ4_K; Name: 'Q4_K'; Rows: 32000; Cols: 4096)
  );

var
  Seed: UInt32 = 987654321;

function NextByte: Byte;
begin
  Seed := Seed * 1664525 + 1013904223;
  Result := Byte((Seed shr 16) and $FF);
end;

function HalfOf(V: Single): Word;
var
  U, S, M: UInt32;
  E: Integer;
begin
  U := PUInt32(@V)^;
  S := (U shr 31) and 1;
  E := Integer((U shr 23) and $FF) - 127 + 15;
  M := U and $7FFFFF;
  if E <= 0 then
    Result := Word(S shl 15)
  else if E >= 31 then
    Result := Word((S shl 15) or (UInt32($1F) shl 10))
  else
    Result := Word((S shl 15) or (UInt32(E) shl 10) or (M shr 13));
end;

{ Builds a tensor whose payload bytes are pseudo-random -- that exercises
  every nibble and bit position -- while the f16 scale fields are forced to
  small sane values so nothing becomes NaN or Inf. }
procedure BuildTensor(var T: TQTensor; Typ: TGgmlType; Rows, Cols: Integer);
var
  R, I, B, NB, NSB: Integer;
  P: PByte;
begin
  T.Typ := Typ;
  T.Rows := Rows;
  T.Cols := Cols;
  SetLength(T.Data, T.TotalBytes);
  for R := 0 to Rows - 1 do
  begin
    P := PByte(T.Data) + Int64(R) * T.RowBytes;
    case Typ of
      gtF32:
        for I := 0 to Cols - 1 do
          PSingle(P)[I] := Cos(I * 0.11 + R) * 0.8;
      gtF16:
        for I := 0 to Cols - 1 do
          PWord(P)[I] := HalfOf(Cos(I * 0.11 + R) * 0.8);
      gtQ8_0:
        begin
          NB := Cols div QK;
          for B := 0 to NB - 1 do
          begin
            PWord(P + B * 34)^ := HalfOf(0.021);
            for I := 0 to QK - 1 do
              (P + B * 34 + 2)[I] := NextByte;
          end;
        end;
      gtQ4_0:
        begin
          NB := Cols div QK;
          for B := 0 to NB - 1 do
          begin
            PWord(P + B * 18)^ := HalfOf(0.033);
            for I := 0 to (QK div 2) - 1 do
              (P + B * 18 + 2)[I] := NextByte;
          end;
        end;
      gtQ4_1:
        begin
          NB := Cols div QK;
          for B := 0 to NB - 1 do
          begin
            PWord(P + B * 20)^ := HalfOf(0.017);
            PWord(P + B * 20 + 2)^ := HalfOf(-0.25);
            for I := 0 to (QK div 2) - 1 do
              (P + B * 20 + 4)[I] := NextByte;
          end;
        end;
      gtQ4_K:
        begin
          NSB := Cols div QK_K;
          for B := 0 to NSB - 1 do
          begin
            PWord(P + B * 144)^ := HalfOf(0.0012);
            PWord(P + B * 144 + 2)^ := HalfOf(0.0007);
            for I := 0 to 139 do
              (P + B * 144 + 4)[I] := NextByte;
          end;
        end;
      gtQ5_K:
        begin
          NSB := Cols div QK_K;
          for B := 0 to NSB - 1 do
          begin
            PWord(P + B * 176)^ := HalfOf(0.0009);
            PWord(P + B * 176 + 2)^ := HalfOf(0.0005);
            for I := 0 to 171 do
              (P + B * 176 + 4)[I] := NextByte;
          end;
        end;
      gtQ6_K:
        begin
          NSB := Cols div QK_K;
          for B := 0 to NSB - 1 do
          begin
            for I := 0 to 207 do
              (P + B * 210)[I] := NextByte;
            PWord(P + B * 210 + 208)^ := HalfOf(0.00045);
          end;
        end;
    end;
  end;
end;

function ArgValue(const Name, Def: string): string;
var
  I: Integer;
begin
  Result := Def;
  for I := 1 to ParamCount - 1 do
    if SameText(ParamStr(I), '--' + Name) then
      Exit(ParamStr(I + 1));
end;

function HasArg(const Name: string): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if SameText(ParamStr(I), '--' + Name) then
      Exit(True);
end;

procedure SetEnvVar(const Name, Value: string);
begin
{$IFDEF MSWINDOWS}
  Winapi.Windows.SetEnvironmentVariable(PChar(Name), PChar(Value));
{$ELSE}
  Posix.Stdlib.setenv(PAnsiChar(AnsiString(Name)),
    PAnsiChar(AnsiString(Value)), 1);
{$ENDIF}
end;

procedure Benchmark(Iters: Integer);
const
  { Checking every row of a 32000-row tensor against a dequantize-and-dot
    reference costs more than the whole benchmark. A sample is enough here --
    the exhaustive check is the self-test's job. }
  ERR_ROWS = 256;
var
  C: Integer;
  T: TQTensor;
  X, YCpu, YGpu, Row: TArray<Single>;
  SW: TStopwatch;
  I, R, NErr: Integer;
  CpuMs, GpuMs, Bytes: Double;
  Acc, Mag, Rel, WorstGpu, WorstCpu: Double;
  Ok: Boolean;
begin
  Writeln;
  Writeln('MatVec benchmark  (', Iters, ' iterations per case)');
  Writeln('Error columns are relative to sum|w*x| and measured against a');
  Writeln('dequantize-then-dot reference, sampled over the first ', ERR_ROWS,
          ' rows.');
  Writeln('cpu-err is nonzero for Q4_0/Q4_1/Q8_0 because the CPU kernel');
  Writeln('quantizes the activation to int8; the GPU kernel does not.');
  Writeln;
  Writeln('shape                      type    CPU ms    GPU ms   speedup   ',
          'GPU GB/s   gpu-err   cpu-err');
  Writeln(StringOfChar('-', 108));

  for C := Low(CASES) to High(CASES) do
  begin
    BuildTensor(T, CASES[C].Typ, CASES[C].Rows, CASES[C].Cols);
    SetLength(X, T.Cols);
    for I := 0 to T.Cols - 1 do
      X[I] := Sin(I * 0.37) * 1.7 - 0.4;
    SetLength(YCpu, T.Rows);
    SetLength(YGpu, T.Rows);
    SetLength(Row, T.Cols);

    { warm up both paths: the GPU run also pays the one-off VRAM upload }
    T.MatVecCpu(@YCpu[0], @X[0]);
    Ok := VulkanMatVec(T, @YGpu[0], @X[0]);

    SW := TStopwatch.StartNew;
    for I := 1 to Iters do
      T.MatVecCpu(@YCpu[0], @X[0]);
    CpuMs := SW.Elapsed.TotalMilliseconds / Iters;

    if Ok then
    begin
      SW := TStopwatch.StartNew;
      for I := 1 to Iters do
        VulkanMatVec(T, @YGpu[0], @X[0]);
      GpuMs := SW.Elapsed.TotalMilliseconds / Iters;
    end
    else
      GpuMs := 0;

    { Reference: dequantize the row and dot it in double. Independent of both
      timed kernels, and free of the CPU path's activation quantization. }
    WorstGpu := 0;
    WorstCpu := 0;
    NErr := Min(T.Rows, ERR_ROWS);
    for R := 0 to NErr - 1 do
    begin
      T.DequantRow(R, @Row[0]);
      Acc := 0;
      Mag := 0;
      for I := 0 to T.Cols - 1 do
      begin
        Acc := Acc + Double(Row[I]) * X[I];
        Mag := Mag + Abs(Double(Row[I]) * X[I]);
      end;
      Mag := Max(Mag, 1.0E-30);
      Rel := Abs(Acc - YCpu[R]) / Mag;
      if Rel > WorstCpu then
        WorstCpu := Rel;
      if Ok then
      begin
        Rel := Abs(Acc - YGpu[R]) / Mag;
        if Rel > WorstGpu then
          WorstGpu := Rel;
      end;
    end;

    Bytes := T.TotalBytes;
    if Ok then
      Writeln(Format('%6d x %-6d %8.1f MB  %-6s %8.2f  %8.2f  %7.1fx  %8.1f  %.2e  %.2e',
        [T.Rows, T.Cols, Bytes / (1024 * 1024), CASES[C].Name, CpuMs, GpuMs,
         CpuMs / Max(GpuMs, 1.0E-9),
         Bytes / (GpuMs / 1000.0) / (1024 * 1024 * 1024), WorstGpu, WorstCpu]))
    else
      Writeln(Format('%6d x %-6d %8.1f MB  %-6s %8.2f       ---        ---       ---       ---  %.2e  (CPU only)',
        [T.Rows, T.Cols, Bytes / (1024 * 1024), CASES[C].Name, CpuMs,
         WorstCpu]));

    VulkanEvict(T);
    T.Data := nil;
  end;
end;

var
  Info, Report: string;
  Budget, Iters: Integer;
  St: TVulkanStats;

begin
  try
    Writeln('Prism ', PRISM_VERSION, ' - GPU verification and benchmark');
    Writeln;

    if ArgValue('device', '') <> '' then
      SetEnvVar('PRISM_VK_DEVICE', ArgValue('device', ''));
    Budget := StrToIntDef(ArgValue('budget', '0'), 0);
    Iters := StrToIntDef(ArgValue('iters', '20'), 20);

    Writeln('CPU backend: ', Prism.Gpu.Backend.Name);
    Writeln;
    Writeln('Bringing up Vulkan...');

    if not VulkanInit(Budget,
      procedure(S: string)
      begin
        Writeln(S);
      end, Info) then
    begin
      Writeln;
      Writeln('No GPU backend: ', Info);
      Writeln('Everything would run on the CPU. Nothing to benchmark.');
      Halt(2);
    end;

    Writeln;
    Writeln('Device: ', Info);

    { The self-test already ran inside VulkanInit -- Init refuses to activate
      if it fails, so reaching here means the kernels agree with the CPU.
      Run it again explicitly so the report is visible. }
    Writeln;
    if VulkanSelfTest(Report) then
      Writeln('Kernel self-test PASSED: ', Report)
    else
    begin
      Writeln('Kernel self-test FAILED: ', Report);
      Writeln('The GPU path is NOT trustworthy. Stopping.');
      Halt(3);
    end;

    if not HasArg('verify-only') then
      Benchmark(Iters);

    St := VulkanStats;
    Writeln;
    Writeln('Final state: ', St.Describe);
  except
    on E: Exception do
    begin
      Writeln('ERROR: ', E.ClassName, ': ', E.Message);
      Halt(1);
    end;
  end;
end.
