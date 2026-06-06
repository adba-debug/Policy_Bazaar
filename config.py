"""
Centralised hyperparameter configuration.
All magic numbers live here so every other module
can simply `from config import Config`.
"""


class Config:
    # Environment
    ENV_NAME        = "ALE/Boxing-v5"
    FRAME_STACK     = 4
    FRAME_SIZE      = 84

    # Network
    LEARNING_RATE   = 1e-4    # ⬆️ Increase slightly from 5e-5 (Faster weight updates)
    BATCH_SIZE      = 128     # Keep at 128 (Protects gradients from exploding at higher LR)
    GAMMA           = 0.95   # ⬇️ Lower from 0.99 (Forces absolute short-term aggression)

    # Epsilon schedule
    EPS_START       = 1.0
    EPS_END         = 0.02    # ⬇️ Lower from 0.10 (Forces pure exploitation at the end)
    EPS_DECAY_STEPS = 40_000  # ⬇️ Crash exploration faster (Gives agent more time to practice winning)

    # Replay buffer
    BUFFER_SIZE     = 40_000
    PER_ALPHA       = 0.6     # Keep PER active to hunt high-loss frames
    PER_BETA_START  = 0.4
    PER_BETA_END    = 1.0
    PER_BETA_STEPS  = 150_000 # ⬇️ Scale down to match your total step timeline
    PER_EPSILON     = 1e-6

    # Training
    TARGET_UPDATE   = 1_000   # ⬇️ Update faster (Forces target network to keep up with aggressive policy)
    TRAIN_START     = 5_000
    TOTAL_STEPS     = 500_000
    SAVE_EVERY      = 50_000
    LOG_EVERY       = 1_000