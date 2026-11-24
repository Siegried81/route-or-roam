# Running agents in a sandbox

Two kinds of agent touch this repository: the agent under evaluation (the
ReAct system in `rr/`), and the coding agents that help write the code. The
long evaluation runs and any unattended coding session happen in a Docker
sandbox, a microVM with its own kernel, filesystem, network and Docker Engine,
on the laptop or in Docker's cloud. Inside it the agent can install packages,
start services and build images without asking at each step.

A sandbox lowers the risk; it does not remove it. The realistic failures here
are not a VM escape but three quieter ones:

- a key or token present in the sandbox leaks;
- the agent writes a file that a trusted tool outside the sandbox later runs,
  loads or scans (a git hook, an editor task, a test fixture, a CI workflow);
- the agent installs a compromised dependency.

## In place

1. **No important secret inside.** No cloud credentials, no production token,
   no personal key; only the Groq keys a run needs, one per worker.
2. **Tool allowlist.** The agent under test can only call the tools declared
   in its Pydantic schema; anything else is refused.
3. **Pinned dependencies.** Installs go through `requirements.lock` and
   `npm ci`; Dependabot keeps the pins moving.
4. **No Docker socket in an agent's container.** Mounting
   `/var/run/docker.sock` gives the container control of the host's Docker.
   `docker-compose.yml` mounts only data volumes, and the image runs as an
   unprivileged user with no `.env` baked in.
5. **Portfolio code only.** Nothing confidential, no client data and no
   third-party repository is given to an unattended agent.
6. **Short-lived and logged.** A sandbox exists for one task and is stopped
   when it ends; the agent's log is kept. A stopped sandbox also stops the
   bill.
7. **Tools up to date.** Docker Desktop and the sandbox CLI are updated before
   a long run.

## Not yet

- **Network allowlist.** The sandbox's network is open today; only the agent's
  tools are restricted. The next step is a network policy that allows the
  package indexes (PyPI, npm), the model API and GitHub, nothing else.
- **Work back through a pull request.** Work comes back by syncing the
  sandbox's workspace. The next step is a pull request on GitHub, read with
  extra care on the files that execute on their own: `.git/hooks/`,
  `.vscode/`, `conftest.py`, `setup.py`, `pyproject.toml`, `Makefile`,
  `Dockerfile`, `docker-compose*.yml`, `.github/workflows/`, and the scripts
  in `web/package.json`.
- **Keys scoped to the run.** The Groq keys are the everyday ones. The next
  step is a key per run with the lowest limits that let it finish, revoked
  when the run is over.

## How it fits the agent's own guards

The agent under test already has guards inside the program: the tool
allowlist, a whitelisted calculator, a fence around retrieved text, a budget
of 8 calls and 12,000 tokens, and human approval before it writes a file
([deep dive](technical_deep_dive.md)). Those limit what the model can do
through its tools. The sandbox limits what anything running in the process can
do to the machine around it. Neither replaces the other.
