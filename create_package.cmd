echo OFF
echo[
echo[
echo                   ###########################
echo                   ##   Building TheaNet    ##
echo                   ###########################
echo[
echo[

dotnet build -c Release

echo TheaNet was built in release configuration, ready to create package
pause
echo[
echo[
echo                 ##################################
echo                 ##   Building TheaNet Package   ##
echo                 ##################################
echo[
echo[

nuget pack Proxem.TheaNet/nuspec/Proxem.TheaNet.nuspec -Symbols -OutputDirectory Proxem.TheaNet/nuspec/

echo Package was created in folder Proxem.TheaNet/nuspec/
pause