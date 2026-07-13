"""
Few-shot examples used to steer the clause-extraction prompt (bonus:
"Experiment with few-shot examples to improve clause extraction").

Each example is a (contract_excerpt, expected_json_string) pair. These are
injected into the prompt as prior user/assistant turns so the model has a
concrete pattern to imitate, which noticeably reduces over-verbose or
hallucinated extractions compared to zero-shot prompting on CUAD-style
dense legal text.
"""

FEW_SHOT_EXAMPLES = [
    {
        "contract_excerpt": (
            "This Agreement shall commence on the Effective Date and continue "
            "for an initial term of three (3) years. Either party may terminate "
            "this Agreement for convenience upon ninety (90) days' prior written "
            "notice to the other party. Either party may terminate immediately "
            "upon a material breach by the other party that remains uncured for "
            "thirty (30) days after written notice of such breach. Each party "
            "agrees to hold the other party's Confidential Information in strict "
            "confidence and not disclose it to any third party without prior "
            "written consent, except as required by law. This obligation survives "
            "termination of the Agreement for a period of five (5) years. "
            "Neither party's aggregate liability under this Agreement shall "
            "exceed the total fees paid in the twelve (12) months preceding the "
            "claim, except for liability arising from breach of confidentiality "
            "or gross negligence."
        ),
        "expected_json": {
            "termination_clause": (
                "Either party may terminate for convenience with 90 days' written "
                "notice, or immediately for uncured material breach after 30 days' "
                "notice."
            ),
            "confidentiality_clause": (
                "Each party must keep the other's Confidential Information "
                "confidential and not disclose it without written consent, "
                "except as required by law; obligation survives termination "
                "for 5 years."
            ),
            "liability_clause": (
                "Aggregate liability is capped at fees paid in the preceding 12 "
                "months, with carve-outs for breach of confidentiality and gross "
                "negligence."
            ),
        },
    },
    {
        "contract_excerpt": (
            "This Non-Disclosure Agreement governs the exchange of proprietary "
            "information between the parties for the purpose of evaluating a "
            "potential business relationship. The Receiving Party shall use the "
            "Disclosing Party's Confidential Information solely for the "
            "Purpose and shall protect it using the same degree of care it uses "
            "for its own confidential information, but no less than reasonable "
            "care. This Agreement contains no provisions regarding termination "
            "or limitation of liability."
        ),
        "expected_json": {
            "termination_clause": "",
            "confidentiality_clause": (
                "The Receiving Party must use Confidential Information only for "
                "the stated evaluation purpose and protect it with at least "
                "reasonable care."
            ),
            "liability_clause": "",
        },
    },
]
