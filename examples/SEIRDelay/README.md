Simple SEIR model with delayed detections.

There are three different config files:
- `config.yaml`: simulates the model and uses the observations for inference

- `config_flu2009`: uses `data/flu2009_incidence.csv`, which is the daily incidence of acute respiratory illness amongst children in a school during the 2009 H1N1 pandemic. Small number of cases. Based on [https://cran.r-project.org/web/packages/EpiEstim/vignettes/demo.html](), which provides some $R_0$ estimates to compare with.

- `config_epinow`: uses `data/epinow_example.csv` which is a toy(?) dataset of notifications. The case numbers are very high, so the particle filter is probably a no-go for this one. Based on [https://epiforecasts.io/EpiNow2/articles/estimate_infections_workflow.html](), which accounts for reporting delays. The waiting time parameters in the config are set by taking the means of the relevant parameter distributions in the EpiNow2 example. The article also provides $R_0$ estimates to compare to.



