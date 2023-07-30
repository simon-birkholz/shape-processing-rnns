# Scripts

Contains the scripts used to run this on the bwUniCluster of the KIT.
To start a single job:
```
bash bwcluster_init.sh 1 name_on_wand_dashbord
```

To start multiple jobs that can run in parallel:
```
bash bwcluster_init.sh 10 name_on_wand_dashbord
```

To a chain job that runs one after another, and reloads the weights of the previous run:
```
bash bwcluster_sweep_init.sh 10 name_on_wand_dashbord
```