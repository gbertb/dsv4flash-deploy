The purpose for this project is to deploy Deepseek v4 Flash on 4x a100 GPU (ampere) on Modal labs architecture.

RULES:
## Modal container cleanup
After any `modal run`, `modal deploy`, or Modal debugging session, verify that no runaway containers are still running.
Check running containers:
```bash
modal container list

Check running apps/deployments:

modal app list

Inspect logs if a container looks hung or errored:

modal container logs <container-id> --tail 100
modal container logs <container-id> -f

Stop a stuck container:

modal container stop <container-id> --yes

Stop an app and terminate its running containers:

modal app stop <app-id-or-name> --yes

Rule: never leave a Modal debugging session without checking container/app status. If anything is hung, errored, or no longer needed, stop it immediately to avoid wasting credits.

After each deployment, create a new CHANGELOG_[deployment_config_name].md and update its status as you debug and try new approaches.
