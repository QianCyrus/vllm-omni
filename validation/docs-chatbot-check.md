# Documentation chatbot preflight

Checked at: 2026-09-25 16:55 UTC (2026-09-26 00:55 Asia/Shanghai).

Status: **NOT COMPLETED. No question was submitted and no chatbot answer was received.**

## Required check

The repository's `.github/ISSUE_TEMPLATE/700-performance-discussion.yml`
requires searching relevant issues and asking the documentation chatbot before
submitting. The checkbox must remain unchecked until the chatbot interaction
actually succeeds, or the issue must explicitly explain why this step could not
be completed.

Requested documentation URL: https://vllm-omni.readthedocs.io/

The public web reader successfully opened that URL and reported a redirect to:
https://docs.vllm.ai/projects/vllm-omni/en/latest/

## Intended question

> Does vLLM-Omni already support decoding a batch of complete images by splitting
> the batch dimension across the existing DiT worker GPUs, using AutoencoderKL
> or AutoencoderKLFlux2? How would that differ from VAE tile/patch parallelism
> and the spatial_shard_height or spatial_shard_width modes? I am interested in
> preserving native complete-image decode and returning images in their
> original batch order.

This question contains no private paths, credentials, or unpublished run data.

## Actual result

The Browser skill was read and its documented browser runtime setup was used.
Browser initialization failed before navigation with:

```text
Cannot find module '.../browser/26.908.40834/scripts/browser-service.mjs'
imported from '.../trusted-worker.js'
```

The installed skill and client path referenced browser version `26.917.71314`;
the missing internal service path referenced the older `26.908.40834` version.
No browser tab was opened by this attempt. No chatbot button, input, submission,
or response could be inspected. This is a local browser runtime failure, not
evidence that the website's chatbot is unavailable to all users.

Primary-source fallback checks:

- The public homepage was readable through the web tool. Its extracted text
  contained normal project documentation and no chatbot answer.
- A direct public HTML fetch returned HTTP 403. It was not retried with altered
  identity, credentials, or a workaround.
- The local `mkdocs.yml` lists memory, mathjax, mermaid, edit/feedback, and
  Slack/forum scripts; a narrow search of local docs overrides/assets found no
  chatbot, Kapa, Inkeep, chat-widget, or DocsBot reference. This is not proof
  about dynamically injected widgets on the deployed site.
- A fresh `gh issue list` search for `"VAE" "batch"` failed with a GitHub API
  TLS handshake timeout. Public GitHub search queries were then run for VAE
  batch parallel decode. They returned repository code/docs; they did not
  produce an actual chatbot response. Earlier duplicate research remains a
  separate check and must not be presented as this chatbot interaction.

## Submission wording

If submission proceeds while this remains unresolved, an honest note is:

> I searched the existing VAE parallelism code, documentation, and related PRs.
> I could not complete the documentation chatbot check because the available
> browser runtime failed before opening the widget. I have not marked that
> check as completed.

No GitHub issue, PR, comment, or other public post was made during this check.

## One bounded Computer Use fallback

After the Browser setup failure, the Computer Use skill was read and its
documented `@oai/sky` bootstrap was attempted once through the JavaScript tool.
The first read-only operation was listing available apps. It failed with:

```text
Sky Computer Use native pipe startup failed
```

No app state could be read and no browser navigation, typing, or submission
occurred. No plugin files, profiles, permissions, or system settings were
modified. No authentication or HTTP 403 restriction was bypassed. The fallback
stopped after this failure; the chatbot check remains **not completed**.
