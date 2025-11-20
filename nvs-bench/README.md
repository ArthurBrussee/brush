# Instructions

See `nvs-bench/README.md` for more detailed instructions. This file is meant to be a sparknotes / reference after using the `git subtree` command to copy the boilerplate files into your method's repo

# Installation
Copying these files into a new method's repo with `git subtree`
```
git remote add nvs-bench https://github.com/N-Demir/nvs-bench.git && git subtree add --prefix=nvs-bench nvs-bench boilerplate-branch --squash
```

Then, fill out:
- `image.py`
- `eval.sh`

I would first test with 10 training iterations to make sure the method integration works end to end before trying to run on the whole benchmark.

# Running
To run:
`modal run -m nvs-bench.runner --data mipnerf360/kitchen`

and then to run on the full benchmark:
`modal run -m nvs-bench.runner`

# Evaluate