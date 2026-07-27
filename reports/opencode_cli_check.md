# OpenCode CLI check

Command executed from the repository root:

```powershell
Get-Command opencode
opencode --version
opencode --help
```

Result:

- Available: yes
- Executable: `C:\Users\10User\.opencode\bin\opencode.exe`
- Version reported: `1.15.7`
- CLI exposes `opencode run`, `opencode agent`, and MCP commands.

The execution used the local CLI for this read-only availability/help check. No external API, data upload, or source-file mutation was performed.

A bounded read-only review was also attempted:

```powershell
opencode run --pure --agent reviewer --format json --dir C:\Users\10User\Documents\khwarsimi ...
```

It exceeded the 60-second limit and returned exit code `124` without changing files. No OpenCode review verdict is claimed.

For the CatBoost v2 rerun, the local CLI was rechecked with `opencode --version` and returned `1.15.7`. The model execution itself remained local Python/CatBoost; no external API or data source was used.
