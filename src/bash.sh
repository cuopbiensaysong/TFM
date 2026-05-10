
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

# #  Overfit 
# python draw_trajectory.py \
#   --config ../outputs/2026-05-05/10-13-35/.hydra/config.yaml \
#   --ckpt ../outputs/2026-05-05/10-13-35/checkpoints/Noise_MLP_Cond_memory_Module_ODE_Memory_3_eICU_DataModule_sepsis/best_model.ckpt \
#   --out-dir ../outputs/2026-05-05/10-13-35/trajectory_plots_ode_t_span_points_100 \
#   --split test \
#   --ode-t-span-points 10 \
#   --seed 42


# python draw_trajectory_compare.py \
#   --config ../outputs/2026-05-03/05-43-13/.hydra/config.yaml \
#   --ckpt ../outputs/2026-05-03/05-43-13/checkpoints/MLP_Cond_memory_Module_ODE_Memory_3_eICU_DataModule_sepsis/best_model.ckpt \
#   --out-dir ../outputs/2026-05-03/05-43-13/trajectory_plots_compare \
#   --split test \
#   --seed 42 \
#   --num-samples 3 \
#   --ode-t-span-points 10

conda activate ti-env
HYDRA_FULL_ERROR=1 python src/main.py 