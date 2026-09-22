# Privacy approach

## Threat model

UK bank transaction descriptions routinely contain personal data: person names
(`BACS JOHN SMITH …`), sort codes (`20-45-67`), account numbers (`12345678`),
email addresses, phone numbers, and address fragments. Sending these to a
hosted LLM API would create an unnecessary data transfer and third-party
processing risk. This project therefore runs everything locally and treats the
raw description as sensitive from the moment it enters the system.

## Controls implemented

1. **Redact first.** The PII redactor adapter runs before preprocessing,
   persistence, training, evaluation exports, and error reporting.
2. **Canonical placeholders.** `[PERSON]`, `[SORT_CODE]`, `[ACCOUNT_NUMBER]`,
   `[EMAIL]`, `[PHONE]`, `[ADDRESS]` (plus `[CARD_NUMBER]`, `[IBAN]`).
3. **Upstream integration.** The vendored
   [UK-PII-Detector-Redactor](https://github.com/EmmaExcel/UK-PII-Detector-Redactor)
   (Presidio + spaCy + UK NHS/NINO/postcode/phone recognisers) is used with its
   real interface. A deterministic UK banking regex backend covers sort codes
   and account numbers that the upstream project does not detect.
4. **Fail safe.** If redaction fails, the service returns a controlled
   `[REDACTION_ERROR]` and never logs the raw text.
5. **Persistence.** Only `redacted_description` and `cleaned_description` are
   written to the database. Feedback accepts and stores redacted text only.
6. **Separation.** Raw ingestion files live in `data/raw`; the processed
   training dataset is saved without a `raw_description` column.
7. **Secrets.** All secrets come from environment variables. `.env` is
   gitignored; `.env.example` contains no credentials.
8. **Deletion.** `DELETE /data/{subject_id}` and
   `database.deletion.delete_subject_data()` / `delete_transaction_data()`
   implement application-level erasure for a user or transaction identifier.
9. **Logs.** Application logs only ever include cleaned/redacted text after the
   redaction stage; tests assert this property.

## Important caveat

**Pseudonymised data can still be personal data.** Redacted text such as
`BACS [PERSON] [SORT_CODE] [ACCOUNT_NUMBER] RENT SEPTEMBER` may remain
identifiable when combined with metadata (amount, timestamp, account context),
and the redactor itself is not perfect. Organisations processing real data must
complete a full data-protection assessment, obtain informed consent where
required, minimise retention, and apply appropriate organisational and
technical measures. This project is a proof of concept, not a compliance
statement.

## Recommended before any real-data use

- Complete a DPIA (data protection impact assessment).
- Obtain informed user consent for collection and processing.
- Replace synthetic fixtures with a consented, documented ingestion pipeline.
- Add a human review workflow for `requires_review=true` predictions.
- Run the PII stack with the presidio/spaCy optional dependencies enabled and
  extend the recognisers for your data.
- Verify deletion flows and backup/retention policies end-to-end.
