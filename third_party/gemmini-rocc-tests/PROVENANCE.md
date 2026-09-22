# Where these files came from

Vendored, not a submodule: upstream stopped moving, so tracking it bought a
dirty working tree and a patch to apply after every checkout, and nothing else.

| | |
|---|---|
| `include/` | https://github.com/ucb-bar/gemmini-rocc-tests `7c540b3adf1b86ad93d07f893abe3a73489b568e` (branch `dev`) |
| `rocc-software/` | https://github.com/ibm/rocc-software `fddb795a0b52e82f8f4ce9ead9b1428440a62ab0` |

Only the headers are kept; the tests, build system and data collection the
upstream repository also carries are not used here.

`LICENSE` is upstream's (BSD 3-clause, Boston University), and
`rocc-software/LICENSE` is that project's own. Both apply to the files beside
them.

**The commit that added this directory is the untouched upstream.** What this
project changed is the commit immediately after it, so

    git log --oneline -- third_party/gemmini-rocc-tests
    git diff <the vendoring commit> HEAD -- third_party/gemmini-rocc-tests

is the whole delta against upstream, for as long as the repository exists. Keep
it that way: land a change to these headers as its own commit with a reason, the
way the rest of this repository does.
