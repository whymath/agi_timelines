from datetime import datetime

# From https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/
# Reliability at 50% success
# Note that 'reliability' is often confused - see https://x.com/peterwildeford/status/1967963942589747557
model_data = {
    'claude_3p7': {
        'name': 'Claude 3.7 Sonnet',
        'launch_date': datetime(2025, 2, 24),
        'performance_50p': 54/60
    },
    'o3': {
        'name': 'o3',
        'launch_date': datetime(2025, 4, 16),
        'performance_50p': 1 + 32/60
    },
    'claude_4': {
        'name': 'Claude 4',
        'launch_date': datetime(2025, 5, 22),
        'performance_50p': 1 + 8/60
    },
    'gpt5': {
        'name': 'GPT5',
        'launch_date': datetime(2025, 8, 7),
        'performance_50p': 2 + 17/60
    },
    'claude_4p5': {
        'name': 'Claude 4.5 Sonnet',
        'launch_date': datetime(2025, 9, 29),
        'performance_50p': None # not yet known
    }
}
