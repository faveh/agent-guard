import httpx

VIOLATION = {"raw_label": "violation", "status": "unsure", "prediction_set": ["violation", "supported"],
             "probabilities": {"supported": 0.05, "violation": 0.9, "insufficient_information": 0.04,
                               "inconsistent_context": 0.01},
             "timing_ms": {"model": 950.0}, "usage": {"input_tokens": 120}}


class FakeHttp:
    """Stands in for httpx.Client: returns the queued (status, json) replies in order and records each post."""

    def __init__(self, replies=None):
        self.replies = list(replies or [(200, VIOLATION)])
        self.posts = []

    def post(self, url, json):
        self.posts.append(json)
        status, body = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))
