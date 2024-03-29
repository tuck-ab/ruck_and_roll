# Automatic Detection and Recording of Key Events in Rugby Games

## Members

* Xander Tuck
* Vita Bush
* Rebecca French
* Joe Owen
* Thomas Newton

## Supervisor

* Arshad Jhumka

## Running the Code

All of the code has been developed as a module. This means the `-m` flag will need
to be added if running the code from the command line. To find instructions on how
to use the modules use `--help`. For the game label model:

```
(venv) python -m game_label_model --help
```

For the labelling tool:

```
(venv) python -m label_tool --help
```

For instructions on how to setup and install the software, see the 
**Virtual Environments and Dependencies** section

## Virtual Environments and Dependencies

It is recommended to use a virtual environment when developing the software
to manage the dependencies. A virtual environment acts as a fresh install
of python with no downloaded libraries. Libraries can be installed to
this virtual environment in the same way using pip, but they are kept
separate from any other libraries downloaded on any other python installations.

### Creating, activating, and deactivating a virtual environment

A virtual environment can be created using

```
python3 -m venv venv
```

in the project root directory. This will create the virtual environment in
a directory called `venv`. This directory has been added to the `.gitignore`
file so should not be added to any commits.

Once the virtual environment has been created it needs to be activated. 

On Linux and MacOS:

```
source path/to/venv/bin/activate
```

On windows (cmd.exe):

```
path\to\venv\Scripts\activate.bat
```

On windows (powershell):

```
path\to\venv\Scripts\Activate.ps1
```

Once the virtual environment has been activated it is used in the same
way as normal python.

To deactivate the virtual environment:

```
deactivate
```

The virtual environment is all stored in the `venv` directory. To
delete it, just delete that directory.

### Installing and defining dependencies

All of the dependencies can be found in `requirements.txt`. They
can easily be installed using `pip`:

```
pip install --upgrade pip
pip install -r requirements.txt
```

### Running on the DCS machines and compute nodes

To create the virtual environment on the DCS machines, run the following command.

```
python3.9 -m venv venv
```

Once the virtual environment has been created, activating and deactivating is the same as on linux system.

A file called `pip.conf` will need to be added to the `venv` directory (`/venv/pip.conf`) and the following text will need to be added.

```
[install]
user = false
```

The DCS machines automatically add the `--user` flag for pip installs which do not work with virtual environments. Once this config file is added then `pip install` should work as normal.

To run on the compute nodes, first `ssh` into `kudu`. This can be done on the machines by running 

```
ssh kudu
```

on any of the computers in DCS and can also be done remotely directly sshing in.

```
ssh u1234567@kudu.dcs.warwick.ac.uk
```

After logging in, navigate to the directory this README is in and run the sbatch script. Make sure to edit the python commands in the script to run what the correct thing.

```
sbatch submit.sbatch
```

To see how busy the `falcon` node running the python is, use the following command.

```
squeue -p falcon
```
