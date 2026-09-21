import unittest

from news import is_relevant_article, normalize_text


class NewsMatchingTests(unittest.TestCase):
    def test_normalize_text_removes_punctuation(self):
        self.assertEqual(normalize_text("Shopify Inc."), "shopify inc")

    def test_is_relevant_article_matches_ticker_and_company_name(self):
        article = {
            "title": "Shopify earnings beat expectations as revenue climbs",
            "summary": "The company says demand remains solid for its commerce platform.",
            "link": "https://example.com/shopify",
        }
        self.assertTrue(is_relevant_article(article, "SHOP", "Shopify"))

    def test_is_relevant_article_rejects_unrelated_story(self):
        article = {
            "title": "Oil prices rise after supply concerns in the Middle East",
            "summary": "Traders are watching global crude inventories.",
            "link": "https://example.com/oil",
        }
        self.assertFalse(is_relevant_article(article, "SHOP", "Shopify"))


if __name__ == "__main__":
    unittest.main()
