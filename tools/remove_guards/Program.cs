using GBX.NET;
using GBX.NET.LZO;
using GBX.NET.Engines.Plug;

if (args.Length < 2)
{
    Console.Error.WriteLine("Usage: remove_guards <input.Solid.Gbx> <output.Solid.Gbx>");
    Environment.ExitCode = 1;
    return;
}

var inputPath = args[0];
var outputPath = args[1];

if (!File.Exists(inputPath))
{
    Console.Error.WriteLine($"Input file not found: {inputPath}");
    Environment.ExitCode = 2;
    return;
}

Gbx.LZO = new MiniLZO();

var gbx = Gbx.Parse<CPlugSolid>(inputPath);
var solid = gbx.Node;

if (solid.Tree is not CPlugTree rootTree)
{
    Console.Error.WriteLine("No tree found in solid");
    Environment.ExitCode = 3;
    return;
}

var guardNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
{
    "dFRGuard", "dFLGuard", "sFRGuard", "sFLGuard",
    "sRLHub", "sRRHub"
};

Console.WriteLine($"Original children: {rootTree.Children.Count}");
foreach (var child in rootTree.Children)
{
    var tag = guardNames.Contains(child.Name ?? "") ? " [REMOVE]" : "";
    Console.WriteLine($"  {child.Name}{tag}");
}

var filteredChildren = rootTree.Children
    .Where(c => !guardNames.Contains(c.Name ?? ""))
    .ToList();

Console.WriteLine($"\nFiltered children: {filteredChildren.Count}");
Console.WriteLine($"Removed: {rootTree.Children.Count - filteredChildren.Count} guard objects");

rootTree.Children.Clear();
foreach (var child in filteredChildren)
{
    rootTree.Children.Add(child);
}

gbx.Save(outputPath);
Console.WriteLine($"\nSaved to: {outputPath}");
