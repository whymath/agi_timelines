from datetime import datetime

# From https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/
# Reliability data: performance_50p = task length at 50% reliability, performance_80p = task length at 80% reliability (in hours)
# Note that 'reliability' is often confused - see https://x.com/peterwildeford/status/1967963942589747557

model_data = {
    'gpt2': {
        'name': 'GPT‑2',
        'launch_date': datetime(2019, 2, 14),
        'performance_50p': 2 / 3600,
        'performance_80p': 0.1 / 3600
    },
    'gpt3': {
        'name': 'GPT-3',
        'launch_date': datetime(2020, 5, 28),
        'performance_50p': 9 / 3600,
        'performance_80p': 2 / 3600
    },
    'gpt3p5_turbo': {
        'name': 'GPT‑3.5 Turbo',
        'launch_date': datetime(2023, 3, 1),
        'performance_50p': 36 / 3600,
        'performance_80p': 10 / 3600
    },
    'gpt4': {
        'name': 'GPT-4',
        'launch_date': datetime(2023, 3, 14),
        'performance_50p': 6 / 60,
        'performance_80p': 1 / 60
    },
    'gpt4_nov23': {
        'name': 'GPT-4-Nov23',
        'launch_date': datetime(2023, 11, 6),
        'performance_50p': 8 / 60,
        'performance_80p': 1 / 60
    },
    'claude_3_opus': {
        'name': 'Claude 3 Opus',
        'launch_date': datetime(2024, 3, 4),
        'performance_50p': 6 / 60,
        'performance_80p': 1 / 60
    },
    'gpt4o': {
        'name': 'GPT‑4o',
        'launch_date': datetime(2024, 5, 13),
        'performance_50p': 9 / 60,
        'performance_80p': 2 / 60
    },
    'claude_3p5_sonnet_old': {
        'name': 'Claude 3.5 Sonnet (old)',
        'launch_date': datetime(2024, 6, 20),
        'performance_50p': 18 / 60,
        'performance_80p': 3 / 60
    },
    'o1_preview': {
        'name': 'o1 preview',
        'launch_date': datetime(2024, 9, 12),
        'performance_50p': 22 / 60,
        'performance_80p': 4 / 60
    },
    'claude_3p5_sonnet_new': {
        'name': 'Claude 3.5 Sonnet (new)',
        'launch_date': datetime(2024, 10, 22),
        'performance_50p': 28 / 60,
        'performance_80p': 5 / 60
    },
    'o1': {
        'name': 'o1',
        'launch_date': datetime(2024, 12, 5),
        'performance_50p': 39 / 60,
        'performance_80p': 6 / 60
    },
    'claude_3p7_sonnet': {
        'name': 'Claude 3.7 Sonnet',
        'launch_date': datetime(2025, 2, 24),
        'performance_50p': 59 / 60,
        'performance_80p': 15 / 60
    },
    'o3': {
        'name': 'o3',
        'launch_date': datetime(2025, 4, 16),
        'performance_50p': 1 + 45 / 60,
        'performance_80p': 20 / 60
    },
    'claude_4_sonnet': {
        'name': 'Claude 4 Sonnet',
        'launch_date': datetime(2025, 5, 22),
        'performance_50p': 1 + 7 / 60,
        'performance_80p': 16 / 60
    },
    'claude_4_opus': {
        'name': 'Claude 4 Opus',
        'launch_date': datetime(2025, 5, 22),
        'performance_50p': 1 + 19 / 60,
        'performance_80p': 20 / 60
    },
    'gemini_2p5_pro': {
        'name': 'Gemini 2.5 Pro',
        'launch_date': datetime(2025, 6, 5),
        'performance_50p': 39 / 60,
        'performance_80p': 9 / 60
    },
    'grok_4': {
        'name': 'Grok 4',
        'launch_date': datetime(2025, 7, 9),
        'performance_50p': 1 + 50 / 60,
        'performance_80p': 15 / 60
    },
    'claude_4p1_opus': {
        'name': 'Claude 4.1 Opus',
        'launch_date': datetime(2025, 8, 5),
        'performance_50p': 1 + 45 / 60,
        'performance_80p': 21 / 60
    },
    'gpt5': {
        'name': 'GPT5',
        'launch_date': datetime(2025, 8, 7),
        'performance_50p': 2 + 17 / 60,
        'performance_80p': 25 / 60
    },
    'claude_4p5_sonnet': {
        'name': 'Claude 4.5 Sonnet',
        'launch_date': datetime(2025, 9, 29),
        'performance_50p': 1 + 53/60,
        'performance_80p': 20 / 60
    },
    'gpt5.1-codex-max': {
        'name': 'GPT5.1-Codex-Max',
        'launch_date': datetime(2025, 11, 19),
        'performance_50p': 2 + 42 / 60,
        'performance_80p': 31 / 60
    },
}
