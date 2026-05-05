
# # SDE with L1 loss 
# python draw_trajectory.py \
#   --config ../outputs/2026-05-04/11-57-22/.hydra/config.yaml \
#   --ckpt ../outputs/2026-05-04/11-57-22/checkpoints/Noise_MLP_Cond_memory_Module_SDE_Memory_3_eICU_DataModule_sepsis/best_model.ckpt \
#   --out-dir ../outputs/2026-05-04/11-57-22/trajectory_plots \
#   --split test \
#   --seed 42


# #  SDE with MSE loss
# python draw_trajectory.py \
#   --config ../outputs/2026-04-29/03-26-17/.hydra/config.yaml \
#   --ckpt ../outputs/2026-04-29/03-26-17/checkpoints/Noise_MLP_Cond_memory_Module_SDE_Memory_3_eICU_DataModule_sepsis/best_model.ckpt \
#   --out-dir ../outputs/2026-04-29/03-26-17/trajectory_plots_ode_t_span_points_100 \
#   --split train \
#   --ode-t-span-points 100 \
#   --seed 42

#  Overfit 
python draw_trajectory.py \
  --config ../outputs/2026-05-05/10-13-35/.hydra/config.yaml \
  --ckpt ../outputs/2026-05-05/10-13-35/checkpoints/Noise_MLP_Cond_memory_Module_ODE_Memory_3_eICU_DataModule_sepsis/best_model.ckpt \
  --out-dir ../outputs/2026-05-05/10-13-35/trajectory_plots_ode_t_span_points_100 \
  --split test \
  --ode-t-span-points 10 \
  --seed 42
