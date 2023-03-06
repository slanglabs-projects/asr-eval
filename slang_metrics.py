
class SlangMetrics:

    def response_subtract(self, A, B):
        result = []
        for i in A:
            if i not in B:
                result.append(i)
        return result

    def response_add(self, A, B):
        out = []
        for i in A+B:
            if i not in out:
                out.append(i)
        return out

    def slang_accuracy_score(self, A, B):
        A_minus_B = self.response_subtract(A, B)
        B_minus_A = self.response_subtract(B, A)
        A_plus_B = self.response_add(A, B)
        numerator = (len(A_minus_B) + len(B_minus_A))
        denominator = len(A_plus_B)
        if numerator == 0:
            score = 0
        else:
            score = numerator / denominator
        return 1 - score

    def preprocess_entity_response(self, response):
        out = []
        for k, v in response['entities'][0].items():
            if isinstance(v, list):
                for val in v:
                    out.append({k: val})
            else:
                out.append({k: v})
        return out

    def compute_translation_score(self, response_1, response_2):
        response_1 = self.preprocess_entity_response(response_1)
        response_2 = self.preprocess_entity_response(response_2)
        score = self.slang_accuracy_score(response_1, response_2)
        return score

    def compute_asr_score(self, response_1, response_2):
        response_1 = self.preprocess_entity_response(response_1)
        response_2 = self.preprocess_entity_response(response_2)
        score = self.slang_accuracy_score(response_1, response_2)
        return score
