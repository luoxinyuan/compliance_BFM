from pathlib import Path
runs=Path(__file__).resolve().parent.parent/'native-transfer-20261107/runs'
names=['short08','C32_short08','C0_short08','I_short08','P_selected_short08','S_selected_short08']
def path(n):
    return runs/('teacher-coverage-01' if n=='short08' else 'bc-eval-01')/(n+'.json.gz')
