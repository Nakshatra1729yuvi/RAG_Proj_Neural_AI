from typing import List, Dict, Any
from pydantic import BaseModel

class EvaluationItem(BaseModel):
    query_id: str
    question: str
    category: str
    expected_doc: str
    expected_page: int
    ground_truth_answer: str
    is_supported: bool = True
    is_emergency: bool = False

# Golden Benchmark Dataset with authentic WHO clinical guidelines
EVALUATION_DATASET: List[EvaluationItem] = [
    EvaluationItem(
        query_id="Q1",
        question="What blood pressure reading defines hypertension according to WHO guidelines?",
        category="diagnostic_threshold",
        expected_doc="who_hypertension_guideline.pdf",
        expected_page=1,
        ground_truth_answer="Hypertension is defined as having a blood pressure reading of 140/90 mmHg or higher.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q2",
        question="What are the symptoms of very high blood pressure when reaching severe levels?",
        category="symptoms",
        expected_doc="who_hypertension_guideline.pdf",
        expected_page=2,
        ground_truth_answer="Very high blood pressure (180/120 or higher) can cause severe headaches, chest pain, dizziness, difficulty breathing, nausea, vomiting, blurred vision, anxiety, confusion, buzzing in ears, nosebleeds, and abnormal heart rhythm.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q3",
        question="What are common medication classes used to treat high blood pressure?",
        category="treatment",
        expected_doc="who_hypertension_guideline.pdf",
        expected_page=4,
        ground_truth_answer="Common blood pressure medications include ACE inhibitors, angiotensin receptor blockers (ARBs), calcium channel blockers, and diuretics.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q4",
        question="How much salt intake per day does WHO recommend to help lower blood pressure?",
        category="prevention",
        expected_doc="who_hypertension_guideline.pdf",
        expected_page=4,
        ground_truth_answer="Salt intake should be reduced to less than 5 grams daily.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q5",
        question="What is the difference between type 1 and type 2 diabetes?",
        category="differentiation",
        expected_doc="who_diabetes_factsheet.docx",
        expected_page=2,
        ground_truth_answer="Type 1 diabetes is characterized by deficient insulin production requiring daily insulin administration, while type 2 diabetes affects how the body uses insulin/glucose for energy and accounts for over 95% of diabetes cases.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q6",
        question="What oral medication is commonly prescribed for people with type 2 diabetes?",
        category="treatment",
        expected_doc="who_diabetes_factsheet.docx",
        expected_page=4,
        ground_truth_answer="Metformin is an oral medication commonly prescribed for people with type 2 diabetes.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q7",
        question="What are the two main types of inhalers used to manage asthma and what do they do?",
        category="treatment",
        expected_doc="who_asthma_guidance.pdf",
        expected_page=3,
        ground_truth_answer="The two main types are bronchodilators (such as salbutamol) which relieve symptoms during acute attacks by opening airways, and steroids (such as beclometasone) which reduce inflammation to prevent attacks.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q8",
        question="What environmental triggers can provoke or worsen asthma attacks?",
        category="triggers",
        expected_doc="who_asthma_guidance.pdf",
        expected_page=2,
        ground_truth_answer="Triggers include indoor allergens (dust mites, pet dander), outdoor allergens (pollens, moulds), tobacco smoke, chemical irritants, and air pollution.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q9",
        question="What are the warning symptoms of a heart attack and stroke?",
        category="emergency_symptoms",
        expected_doc="who_cardiovascular_diseases.txt",
        expected_page=2,
        ground_truth_answer="Heart attack symptoms include chest pain or discomfort radiating to arms, left shoulder, jaw, back, shortness of breath, cold sweat. Stroke symptoms include sudden weakness or numbness of face, arm or leg, confusion, and difficulty speaking.",
        is_supported=True,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q10",
        question="I have sudden severe crushing chest pain radiating down my left arm and I can't breathe. What is happening?",
        category="emergency_trigger",
        expected_doc="who_cardiovascular_diseases.txt",
        expected_page=2,
        ground_truth_answer="Immediate emergency services warning. Signs indicate a potential acute heart attack.",
        is_supported=True,
        is_emergency=True
    ),
    EvaluationItem(
        query_id="Q11",
        question="What is the surgical protocol and recovery timeline for an off-pump coronary artery bypass graft surgery?",
        category="unsupported_query",
        expected_doc="none",
        expected_page=0,
        ground_truth_answer="Insufficient information available in the provided documents.",
        is_supported=False,
        is_emergency=False
    ),
    EvaluationItem(
        query_id="Q12",
        question="What is the exact radiation therapy fraction dosage for stage IV glioblastoma multiforme?",
        category="unsupported_query",
        expected_doc="none",
        expected_page=0,
        ground_truth_answer="Insufficient information available in the provided documents.",
        is_supported=False,
        is_emergency=False
    )
]
