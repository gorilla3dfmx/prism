program DispatchCost;

{ Trennt die FIXKOSTEN eines GPU-Rechenauftrags (Befehlspuffer schreiben,
  vkQueueSubmit, auf den Fence warten, Ergebnis zurueckkopieren) von der
  eigentlichen Rechenzeit im Shader.

  Dazu laeuft dieselbe MatVec ueber Tensoren von winzig bis gross. Bei einer
  64x64-Matrix ist im Shader praktisch nichts zu tun -- was dann noch an Zeit
  bleibt, ist der Rundlauf. Der Rest der Reihe zeigt, ab wann die Rechenarbeit
  ueberhaupt sichtbar wird.

  Alles in F32: der einfachste Kernel, damit nichts vom Entpacken die Messung
  faerbt. }

{$APPTYPE CONSOLE}
{$POINTERMATH ON}

uses
  System.SysUtils, System.Math, System.Diagnostics,
  Prism.Types in '..\src\Prism.Types.pas',
  Prism.Vector in '..\src\Prism.Vector.pas',
  Prism.Vulkan.Api in '..\src\Prism.Vulkan.Api.pas',
  Prism.Vulkan in '..\src\Prism.Vulkan.pas',
  Prism.Gpu in '..\src\Prism.Gpu.pas';

const
  SHAPES: array [0 .. 8] of record R, C: Integer end = (
    (R: 64;    C: 64),
    (R: 256;   C: 256),
    (R: 512;   C: 1024),
    (R: 1024;  C: 4096),
    (R: 2048;  C: 4096),
    (R: 4096;  C: 4096),
    (R: 8192;  C: 4096),
    (R: 14336; C: 4096),
    (R: 4096;  C: 14336)
  );
  ITERS = 50;

var
  Info: string;
  S, I, J: Integer;
  T: TQTensor;
  X, Y: TArray<Single>;
  W: TStopwatch;
  Ms, PrevMs, PrevRows: Double;
  Ok: Boolean;

begin
  if not TryInitGpuBackend(Info, 0, procedure(Sx: string) begin end) then
  begin
    Writeln('Keine GPU: ', Info);
    Halt(1);
  end;
  Writeln('GPU: ', Info);
  Writeln;
  Writeln('Alle Faelle F32. "ms" ist ein vollstaendiger MatVec-Aufruf,');
  Writeln('also Shader PLUS Rundlauf.');
  Writeln;
  Writeln('    Zeilen x Spalten      MB       ms    ns/Zeile     GB/s');
  Writeln(StringOfChar('-', 62));

  PrevMs := 0;
  PrevRows := 0;
  for S := 0 to High(SHAPES) do
  begin
    T.Typ := gtF32;
    T.Rows := SHAPES[S].R;
    T.Cols := SHAPES[S].C;
    SetLength(T.Data, T.TotalBytes);
    for I := 0 to (Length(T.Data) div 4) - 1 do
      PSingle(@T.Data[0])[I] := 0.001;

    SetLength(X, T.Cols);
    SetLength(Y, T.Rows);
    for I := 0 to T.Cols - 1 do
      X[I] := 0.5;

    { Aufwaermen: erst dabei wandern die Gewichte in den Grafikspeicher. }
    for I := 1 to 3 do
      T.MatVec(@Y[0], @X[0]);

    W := TStopwatch.StartNew;
    for I := 1 to ITERS do
      T.MatVec(@Y[0], @X[0]);
    Ms := W.Elapsed.TotalMilliseconds / ITERS;

    Writeln(Format('%10d x %-6d %7.1f %8.3f %11.1f %8.1f',
      [T.Rows, T.Cols, T.TotalBytes / 1048576, Ms,
       Ms * 1e6 / T.Rows, T.TotalBytes / 1073741824 / (Ms / 1000)]));

    T.Data := nil;
  end;

  Writeln;
  Writeln('GPU: ', GpuDetails);
  ShutdownGpuBackend;
end.
