# README 

This requires a modified version of SatGen from the "apc" branch of `https://github.com/nthu-ga/SatGen`

To hardcode the import path for SatGen (in the absence of APC writing a proper
setup.py for this module...) you can add a file "config.py" in `py/sgarrak`.
The contents of this file should simply be a single variable definition that
points to your SatGen directory, as follows:

```
SATGEN_PATH = '/data/apcooper/sfw/SatGen'
```

Obviously you should substitute your own path!

The file `py/sgarrak/config.py` is deliberately not included in the sgarrak git
repository, so different users can have their own local configuration that
isn't overwritten by pulling new versions from github.
