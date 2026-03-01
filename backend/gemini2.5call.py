#!/usr/bin/env python3
"""Call Gemini / Google Generative API using `prompt.txt`.

This script uses the official Google `google-generativeai` library when
``--google`` is passed or ``GEMINI_API_TYPE=google`` is set. It falls back to a
generic HTTP POST using ``requests`` when not using the Google-specific flow.

Usage examples (PowerShell):
	python .\backend\gemini2.5call.py --google --model "models/text-bison-001" --output resp.json
	python .\backend\gemini2.5call.py --endpoint "https://example.com/generate" --api-key KEY

Environment variables supported:
	GEMINI_ENDPOINT, GEMINI_API_KEY, GEMINI_API_TYPE, GEMINI_MODEL, BEARER_TOKEN
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

BASE_DIR = os.path.dirname(__file__)
PROMPT_PATH = os.path.join(BASE_DIR, "prompt.txt")


def load_prompt(path: str = PROMPT_PATH) -> str:
	with open(path, "r", encoding="utf-8") as f:
		return f.read().strip()


def use_google_genai(prompt: str, api_key: Optional[str], model: Optional[str], temperature: Optional[float] = None):
	try:
		import google.generativeai as genai
	except Exception as e:
		raise RuntimeError(
			"Missing `google-generativeai` package. Install with `pip install google-generativeai`"
		) from e

	# Configure API key
	if not api_key:
		raise RuntimeError("Google API key required (GEMINI_API_KEY or --api-key)")

	# Different versions expose different helpers; try configure()
	try:
		# Newer versions use `genai.configure`
		genai.configure(api_key=api_key)
	except Exception:
		# Some older releases may accept direct assignment
		try:
			genai.api_key = api_key  # type: ignore[attr-defined]
		except Exception:
			pass

	# Choose model and call the most likely generation method available
	model = model or os.environ.get("GEMINI_MODEL") or "models/text-bison-001"

	# Prefer text generation API
	if hasattr(genai, "generate_text"):
		# Typical: genai.generate_text(model=..., prompt=...)
		kwargs = {"model": model, "prompt": prompt}
		if temperature is not None:
			kwargs["temperature"] = temperature
		resp = genai.generate_text(**kwargs)
		return resp

	# Some versions use `generate` with `input` or `prompt`
	if hasattr(genai, "generate"):
		kwargs = {"model": model, "input": prompt}
		if temperature is not None:
			kwargs["temperature"] = temperature
		resp = genai.generate(**kwargs)
		return resp

	# Chat-style interface
	if hasattr(genai, "chat") and hasattr(genai.chat, "create"):
		messages = [{"role": "user", "content": prompt}]
		kwargs = {"model": model, "messages": messages}
		if temperature is not None:
			kwargs["temperature"] = temperature
		resp = genai.chat.create(**kwargs)
		return resp

	raise RuntimeError("Unsupported google-generativeai library version installed")


def call_generic_endpoint(endpoint: str, payload: dict, api_key: Optional[str], bearer_token: Optional[str]):
	try:
		import requests
	except Exception:
		raise RuntimeError("Missing `requests` package. Install with `pip install requests`.")

	headers = {"Content-Type": "application/json"}
	params = None
	if api_key:
		params = {"key": api_key}
	if bearer_token:
		headers["Authorization"] = f"Bearer {bearer_token}"

	resp = requests.post(endpoint, params=params, headers=headers, json=payload, timeout=60)
	try:
		return resp.status_code, resp.json()
	except Exception:
		return resp.status_code, resp.text


def main() -> int:
	parser = argparse.ArgumentParser(description="Call Gemini / Google Generative API using prompt.txt")
	parser.add_argument("--endpoint", help="Fallback generic HTTP endpoint URL to call")
	parser.add_argument("--api-key", help="API key (for Google or generic endpoint)")
	parser.add_argument("--bearer-token", help="Bearer token for Authorization header (generic endpoint)")
	parser.add_argument("--google", action="store_true", help="Use the google-generativeai library")
	parser.add_argument("--model", help="Model name to use with google-generativeai (e.g. models/text-bison-001)")
	parser.add_argument("--temperature", type=float, default=None, help="Temperature for generation (Google only)")
	parser.add_argument("--output", help="Save output to this file")

	args = parser.parse_args()

	endpoint = args.endpoint or os.environ.get("GEMINI_ENDPOINT")
	api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
	bearer = args.bearer_token or os.environ.get("BEARER_TOKEN")
	gemini_type = os.environ.get("GEMINI_API_TYPE", "")

	try:
		prompt = load_prompt()
	except FileNotFoundError:
		print(f"Error: prompt file not found at {PROMPT_PATH}")
		return 3

	# Use Google library when requested explicitly
	use_google = args.google or gemini_type.lower() == "google"

	if use_google:
		try:
			print("Using google-generativeai library...")
			resp = use_google_genai(
				prompt, api_key, args.model or os.environ.get("GEMINI_MODEL", ""), args.temperature
			)
			# Try converting response to JSON-friendly form
			try:
				# many genai responses have .to_dict()
				out = getattr(resp, "to_dict", lambda: resp)()
			except Exception:
				out = resp
			print(json.dumps(out, indent=2, ensure_ascii=False) if not isinstance(out, str) else out)
			if args.output:
				with open(args.output, "w", encoding="utf-8") as f:
					if isinstance(out, (dict, list)):
						json.dump(out, f, indent=2, ensure_ascii=False)
					else:
						f.write(str(out))
			return 0
		except Exception as e:
			print("Google GenAI call failed:", e)
			return 4

	# Fallback to generic HTTP POST
	payload = {"prompt": prompt}
	if not endpoint:
		print("Error: no endpoint provided. Set --endpoint or GEMINI_ENDPOINT env var, or use --google.")
		return 2

	status, result = call_generic_endpoint(endpoint, payload, api_key, bearer)
	print(f"Status: {status}")
	if isinstance(result, dict):
		print(json.dumps(result, indent=2, ensure_ascii=False))
	else:
		print(result)

	if args.output:
		with open(args.output, "w", encoding="utf-8") as out:
			if isinstance(result, dict):
				json.dump(result, out, indent=2, ensure_ascii=False)
			else:
				out.write(str(result))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
