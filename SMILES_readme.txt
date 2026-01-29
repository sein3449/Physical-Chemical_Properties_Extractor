# Unified SMILES GUI (Windows)

SMILES(소분자/폴리머 repeat unit 포함)를 입력하면 RDKit 기반 구조 미리보기, HSP/Polarity 계산, PubChem 물성 수집, (옵션) xTB 계산, (옵션) pGrAdd 열화학 추정을 수행하는 GUI입니다.
PubChem 데이터는 PUG-View REST API를 사용하며 인터넷 연결이 필요합니다.

---

## 실행 전 준비물(제3자 체크리스트)

	1) 필수 설치

- Miniconda/Anaconda 설치(Windows)
- Python은 conda 환경으로 관리하는 것을 권장(특히 RDKit 때문)

	2) conda 환경 만들기 (권장)

- 새 환경 생성/활성화
```bat
conda create -n smiles-gui python=3.11 -c conda-forge rdkit -y
conda activate smiles-gui

	3) 필수 패키지 설치 (이 프로젝트가 import 하는 것들)

아래 패키지들은 코드에서 실제로 import 합니다.
conda install -c conda-forge -y pyside6 pandas requests pubchempy

	4) (옵션) xTB 사용 시

xTB 기능을 켜려면 xtb가 시스템에서 실행 가능해야 합니다(터미널에서 xtb --version이 되어야 함).
conda-forge로 설치 가능:
		conda install -c conda-forge -y xtb

	5) (옵션) pGrAdd 사용 시

pGrAdd 열화학 추정을 켜려면 pgradd가 설치되어야 합니다.
pip install pgradd
환경에 따라 No property sets defined 경고가 나올 수 있으며, 그 경우 pGrAdd 옵션을 끄고 사용하세요.

	6) 네트워크/권한

PubChem 물성 수집은 인터넷 연결이 필요합니다(연구실/기관망에서 외부 HTTPS가 막혀있으면 PubChem 항목이 비게 됨).

---

실행 방법
conda 환경 활성화

conda activate smiles-gui
코드 실행(엔트리포인트)

아래 파일명을 레포에 올린 메인 실행 파일명으로 맞춰서 실행하세요.

python unified_smiles_gui_final_physchem_pubchem_fix.py
자주 나는 문제(간단 해결)
(Windows) xTB 실행 중 cp949/UnicodeDecodeError:

xTB 출력 인코딩과 Windows 기본 인코딩이 충돌할 수 있습니다.

우선 GUI에서 “Run xTB” 옵션을 끄고 사용하거나, xTB 설치/업데이트 후 재시도하세요.

RDKit SMILES Parse Error (C[SO3-] 등):

SMILES 축약 표기 때문에 RDKit 파싱이 실패할 수 있습니다.

가능한 RDKit 호환 표기(예: CS(=O)(=O)[O-])로 바꿔서 입력하세요.

---

참고
RDKit은 conda-forge 설치를 권장합니다.
xTB는 conda-forge에서 conda install xtb로 설치할 수 있습니다.
PubChem PUG-View는 .../rest/pug_view/... 형태의 REST API입니다.
MIT License 기반 모든 저작권리는 POSTECH Sein Chung 의 아래에 있습니다.
2026 Copyright POSTECH Sein Chung All rights reserved.