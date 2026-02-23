.PHONY: ci

ACT ?= act
ACT_ARCH ?= linux/amd64
ACT_JOB ?= check-and-test

ci:
	@command -v $(ACT) >/dev/null || { echo "act command not found"; exit 1; }
	$(ACT) --container-architecture $(ACT_ARCH) -j $(ACT_JOB) --env ACT=true
