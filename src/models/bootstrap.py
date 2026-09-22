
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from preprocessing.features import LABEL2ID, LABELS

_KEYWORDS: List[tuple[str, str]] = [
    ("CASH WITHDRAWAL", "Cash Withdrawal"),
    ("WITHDRAWAL", "Cash Withdrawal"),
    ("ATM", "Cash Withdrawal"),
    ("COUNCIL TAX", "Taxes"),
    ("HMRC", "Taxes"),
    ("TAX", "Taxes"),
    ("RENT", "Rent/Mortgage"),
    ("MORTGAGE", "Rent/Mortgage"),
    ("LANDLORD", "Rent/Mortgage"),
    ("PAYROLL", "Salary"),
    ("SALARY", "Salary"),
    ("WAGES", "Salary"),
    ("TRANSFER", "Transfers"),
    ("SAVINGS", "Transfers"),
    ("ISA", "Transfers"),
    ("PAYPAL", "Subscriptions"),
    ("NETFLIX", "Subscriptions"),
    ("SPOTIFY", "Subscriptions"),
    ("PRIME", "Subscriptions"),
    ("DISNEY", "Subscriptions"),
    ("NOW TV", "Subscriptions"),
    ("NOWTV", "Subscriptions"),
    ("APPLE MUSIC", "Subscriptions"),
    ("APPLE.COM", "Subscriptions"),
    ("GOOGLE STORAGE", "Subscriptions"),
    ("GOOGLE ONE", "Subscriptions"),
    ("ADOBE", "Subscriptions"),
    ("MICROSOFT 365", "Subscriptions"),
    ("DROPBOX", "Subscriptions"),
    ("TFL", "Transport"),
    ("TRANSPORT FOR LONDON", "Transport"),
    ("UBER", "Transport"),
    ("BOLT", "Transport"),
    ("TRAINLINE", "Transport"),
    ("NATIONAL RAIL", "Transport"),
    ("STAGECOACH", "Transport"),
    ("FIRST BUS", "Transport"),
    ("ARRIVA", "Transport"),
    ("BLACK CAB", "Transport"),
    ("GETT", "Transport"),
    ("BP", "Fuel"),
    ("SHELL", "Fuel"),
    ("ESSO", "Fuel"),
    ("TEXACO", "Fuel"),
    ("GULF", "Fuel"),
    ("MURCO", "Fuel"),
    ("PETROL", "Fuel"),
    ("COSTA", "Coffee Shops"),
    ("PRET", "Coffee Shops"),
    ("GREGGS", "Coffee Shops"),
    ("STARBUCKS", "Coffee Shops"),
    ("NERO", "Coffee Shops"),
    ("TESCO", "Groceries"),
    ("SAINSBURY", "Groceries"),
    ("ALDI", "Groceries"),
    ("LIDL", "Groceries"),
    ("ASDA", "Groceries"),
    ("WAITROSE", "Groceries"),
    ("MARKS & SPENCER", "Groceries"),
    ("M&S", "Groceries"),
    ("MORRISONS", "Groceries"),
    ("CO-OP", "Groceries"),
    ("COOP", "Groceries"),
    ("OCADO", "Groceries"),
    ("ICELAND", "Groceries"),
    ("NANDO", "Dining"),
    ("WAGAMAMA", "Dining"),
    ("PIZZA EXPRESS", "Dining"),
    ("ZIZZI", "Dining"),
    ("PREZZO", "Dining"),
    ("FIVE GUYS", "Dining"),
    ("MCDONALD", "Dining"),
    ("KFC", "Dining"),
    ("BURGER KING", "Dining"),
    ("SUBWAY", "Dining"),
    ("WETHERSPOONS", "Dining"),
    ("GREENE KING", "Dining"),
    ("PIZZA HUT", "Dining"),
    ("DOMINO", "Dining"),
    ("HARVESTER", "Dining"),
    ("BRITISH GAS", "Utilities"),
    ("OCTOPUS", "Utilities"),
    ("EDF", "Utilities"),
    ("E.ON", "Utilities"),
    ("EON", "Utilities"),
    ("SCOTTISH POWER", "Utilities"),
    ("THAMES WATER", "Utilities"),
    ("SEVERN TRENT", "Utilities"),
    ("UNITED UTILITIES", "Utilities"),
    ("VIRGIN MEDIA", "Utilities"),
    ("BROADBAND", "Utilities"),
    ("TALKTALK", "Utilities"),
    ("VODAFONE", "Utilities"),
    ("BT", "Utilities"),
    ("AVIVA", "Insurance"),
    ("ADMIRAL", "Insurance"),
    ("DIRECT LINE", "Insurance"),
    ("CHURCHILL", "Insurance"),
    ("HASTINGS", "Insurance"),
    ("PRUDENTIAL", "Insurance"),
    ("LEGAL & GENERAL", "Insurance"),
    ("VITALITY", "Insurance"),
    ("INSURANCE", "Insurance"),
    ("BOOTS", "Healthcare"),
    ("SUPERDRUG", "Healthcare"),
    ("NHS", "Healthcare"),
    ("DENTIST", "Healthcare"),
    ("DENTAL", "Healthcare"),
    ("OPTICIAN", "Healthcare"),
    ("SPECSAVERS", "Healthcare"),
    ("BUPA", "Healthcare"),
    ("PHARMACY", "Healthcare"),
    ("UNIVERSITY", "Education"),
    ("UCL", "Education"),
    ("KINGS COLLEGE", "Education"),
    ("COURSERA", "Education"),
    ("UDEMY", "Education"),
    ("COLLEGE", "Education"),
    ("SCHOOL FEES", "Education"),
    ("NURSERY FEES", "Education"),
    ("JUSTGIVING", "Charity"),
    ("OXFAM", "Charity"),
    ("BRITISH HEART", "Charity"),
    ("CANCER RESEARCH", "Charity"),
    ("RED CROSS", "Charity"),
    ("SAVE THE CHILDREN", "Charity"),
    ("RNLI", "Charity"),
    ("MACMILLAN", "Charity"),
    ("NSPCC", "Charity"),
    ("AGE UK", "Charity"),
    ("GREENPEACE", "Charity"),
    ("BOOKING.COM", "Travel"),
    ("EXPEDIA", "Travel"),
    ("AIRBNB", "Travel"),
    ("BRITISH AIRWAYS", "Travel"),
    ("EASYJET", "Travel"),
    ("RYANAIR", "Travel"),
    ("JET2", "Travel"),
    ("PREMIER INN", "Travel"),
    ("TRAVELODGE", "Travel"),
    ("EUROSTAR", "Travel"),
    ("CINEWORLD", "Entertainment"),
    ("ODEON", "Entertainment"),
    ("VUE", "Entertainment"),
    ("TICKETMASTER", "Entertainment"),
    ("STEAM", "Entertainment"),
    ("PLAYSTATION", "Entertainment"),
    ("XBOX", "Entertainment"),
    ("NINTENDO", "Entertainment"),
    ("AMAZON", "Shopping"),
    ("AMZN", "Shopping"),
    ("EBAY", "Shopping"),
    ("ETSY", "Shopping"),
    ("ARGOS", "Shopping"),
    ("JOHN LEWIS", "Shopping"),
    ("PRIMARK", "Shopping"),
    ("IKEA", "Shopping"),
    ("CURRY", "Shopping"),
    ("NEXT", "Shopping"),
    ("FEE", "Fees"),
    ("OVERDRAFT", "Fees"),
    ("INTEREST CHARGE", "Fees"),
    ("POST OFFICE", "Other"),
    ("GOV.UK", "Other"),
    ("PASSPORT", "Other"),
    ("ROYAL MAIL", "Other"),
    ("DVLA", "Other"),
]


class BootstrapCategoriser:

    name = "bootstrap"
    version = "0.0.0"

    def predict(self, texts: List[str], meta_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        probs = []
        for text, meta in zip(texts, meta_features):
            probs.append(self._score_one(text, meta))
        prob_matrix = np.stack(probs, axis=0)
        topk_indices = np.argsort(-prob_matrix, axis=1)[:, :3]
        topk_probs = np.take_along_axis(prob_matrix, topk_indices, axis=1)
        return {
            "probabilities": prob_matrix,
            "topk_indices": topk_indices,
            "topk_probs": topk_probs,
        }

    def _score_one(self, cleaned_text: str, meta: Dict[str, Any]) -> np.ndarray:
        scores = np.full(len(LABELS), 0.1, dtype=np.float64)
        text = (cleaned_text or "").upper()

        for keyword, label in _KEYWORDS:
            if keyword in text:
                scores[LABEL2ID[label]] += 1.0


        if meta.get("direction") == "credit":
            scores[LABEL2ID["Transfers"]] += 0.4
            scores[LABEL2ID["Salary"]] += 0.4
        if "TRANSFER" in text:
            scores[LABEL2ID["Transfers"]] += 0.6

        total = scores.sum()
        return scores / total


def bootstrap_model_info() -> Dict[str, Any]:
    return {
        "model_name": "bootstrap",
        "version": "0.0.0",
        "supported_labels": LABELS,
        "training_timestamp": None,
        "metrics": {},
    }
