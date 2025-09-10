# README 

This requires a modified version of SatGen from the "apc" branch of `https://github.com/nthu-ga/SatGen`

This code (sgarrak) needs to be able to see the modified version of SatGen on
its pythonpath. The reccomended way to do this is by creating a file
`config.py` in the folder `~/.config/sgarrak`, where `~` is your home
directory. For example, the following bash command does the job:

`touch ~/.config/sgarrak/config.py`

You must then add a single line to this file pointing to the top level of your
SatGen repository:

```
SATGEN_PATH = '/path/to/your/satgen'
```

You should substitute your own path!

If needed, you can put the `config.py` in a different location and set the
environment variable `SGARRAK_CONFIG_DIR = /path/to/config.py`.
