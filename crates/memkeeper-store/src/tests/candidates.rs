//! Tests for candidates operations.

use super::*;

#[test]
fn candidate_submit_list_approve_promotes_to_memory() {
    let path = temp_store_path("candidate_submit_list_approve");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = candidate_submit_request("prefer tabs over spaces in code");
    request.kind = Some("preference".to_string());
    request.source_type = Some("explicit-user".to_string());
    request.rationale = Some("Stated directly in conversation.".to_string());
    let submitted = submit_candidate(&path, &request).expect("submit succeeds");
    assert_eq!(submitted.candidate.status, "pending");
    assert_eq!(submitted.candidate.source_type, "explicit-user");
    assert!(submitted.candidate.resulting_memory_id.is_none());
    let candidate_id = submitted.candidate.id.clone();

    let pending = list_candidates(
        &path,
        &CandidateListRequest {
            status: Some("pending".to_string()),
            space: None,
            limit: 50,
            offset: 0,
        },
    )
    .expect("list pending");
    assert_eq!(pending.total, 1);
    assert_eq!(pending.candidates[0].id, candidate_id);

    let approved = approve_candidate(
        &path,
        &CandidateApproveRequest {
            id: candidate_id.clone(),
            embedding: None,
            embedding_model_id: None,
            dry_run: false,
        },
    )
    .expect("approve succeeds");
    assert_eq!(approved.candidate.status, "approved");
    assert_eq!(approved.memory.status, "active");
    assert_eq!(approved.memory.kind, "preference");
    assert!(approved
        .memory
        .entity_key
        .as_deref()
        .is_some_and(|key| key.starts_with("auto:")));
    assert!(approved
        .memory
        .claim_key
        .as_deref()
        .is_some_and(|key| key.starts_with("auto:")));
    let projected_entities = search_entities(
        &path,
        &EntitySearchRequest {
            entity_key: approved.memory.entity_key.clone(),
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("candidate entity projection search succeeds");
    assert_eq!(projected_entities.results.len(), 1);
    assert_eq!(
        approved.candidate.resulting_memory_id.as_deref(),
        Some(approved.memory.id.as_str())
    );

    let memory = get_memory(
        &path,
        &approved.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: true,
        },
    )
    .expect("get succeeds");
    assert_eq!(memory.content, "prefer tabs over spaces in code");
    assert!(memory
        .source_ref_json
        .as_deref()
        .unwrap_or_default()
        .contains("\"source_type\":\"explicit-user\""));

    let pending_after = list_candidates(
        &path,
        &CandidateListRequest {
            status: Some("pending".to_string()),
            space: None,
            limit: 50,
            offset: 0,
        },
    )
    .expect("list pending after");
    assert_eq!(pending_after.total, 0);

    cleanup_store(&path);
}

#[test]
fn candidate_reject_marks_rejected_with_reason() {
    let path = temp_store_path("candidate_reject");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let submitted =
        submit_candidate(&path, &candidate_submit_request("a noisy guess")).expect("submit");
    let rejected = reject_candidate(
        &path,
        &CandidateRejectRequest {
            id: submitted.candidate.id.clone(),
            reason: Some("duplicate".to_string()),
            dry_run: false,
        },
    )
    .expect("reject succeeds");
    assert_eq!(rejected.candidate.status, "rejected");
    assert_eq!(
        rejected.candidate.decided_reason.as_deref(),
        Some("duplicate")
    );
    assert!(rejected.candidate.decided_at.is_some());

    cleanup_store(&path);
}

#[test]
fn candidate_decision_rejects_non_pending() {
    let path = temp_store_path("candidate_non_pending");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let submitted =
        submit_candidate(&path, &candidate_submit_request("approve me once")).expect("submit");
    let id = submitted.candidate.id.clone();
    approve_candidate(
        &path,
        &CandidateApproveRequest {
            id: id.clone(),
            embedding: None,
            embedding_model_id: None,
            dry_run: false,
        },
    )
    .expect("first approve succeeds");

    let second = approve_candidate(
        &path,
        &CandidateApproveRequest {
            id: id.clone(),
            embedding: None,
            embedding_model_id: None,
            dry_run: false,
        },
    );
    assert!(matches!(second, Err(Error::InvalidRequest { .. })));
    let reject = reject_candidate(
        &path,
        &CandidateRejectRequest {
            id,
            reason: None,
            dry_run: false,
        },
    );
    assert!(matches!(reject, Err(Error::InvalidRequest { .. })));

    cleanup_store(&path);
}

#[test]
fn candidate_submit_validates_enums_and_dry_run() {
    let path = temp_store_path("candidate_validate");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut bad_source = candidate_submit_request("bad source");
    bad_source.source_type = Some("rumor".to_string());
    assert!(matches!(
        submit_candidate(&path, &bad_source),
        Err(Error::InvalidRequest { .. })
    ));

    let mut bad_sensitivity = candidate_submit_request("bad sensitivity");
    bad_sensitivity.sensitivity = Some("top-secret".to_string());
    assert!(matches!(
        submit_candidate(&path, &bad_sensitivity),
        Err(Error::InvalidRequest { .. })
    ));

    let mut dry = candidate_submit_request("not persisted");
    dry.dry_run = true;
    let report = submit_candidate(&path, &dry).expect("dry-run submit succeeds");
    assert!(report.dry_run);
    let all = list_candidates(
        &path,
        &CandidateListRequest {
            status: None,
            space: None,
            limit: 50,
            offset: 0,
        },
    )
    .expect("list all");
    assert_eq!(all.total, 0);

    cleanup_store(&path);
}

#[test]
fn adjudication_guard_decides_ok_degraded_refuse() {
    use crate::{adjudication_guard, AdjudicationGuard};
    // Verdict present -> promote regardless of requirement.
    assert_eq!(adjudication_guard(true, true), AdjudicationGuard::Ok);
    assert_eq!(adjudication_guard(true, false), AdjudicationGuard::Ok);
    // No verdict + required -> fail closed.
    assert_eq!(adjudication_guard(false, true), AdjudicationGuard::Refuse);
    // No verdict + not required -> promote but degraded (warn).
    assert_eq!(
        adjudication_guard(false, false),
        AdjudicationGuard::Degraded
    );
}

#[test]
fn approve_promotes_capture_candidate_with_adjudication_verdict() {
    let path = temp_store_path("candidate_capture_adjudicated");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = candidate_submit_request("fact: the sky was clear on 3 May");
    request.source_type = Some("capture".to_string());
    request.source_json = Some(r#"{"adjudication":{"overall":"clean"}}"#.to_string());
    let submitted = submit_candidate(&path, &request).expect("submit succeeds");

    // A capture candidate carrying an adjudication verdict promotes (guard = Ok).
    let report = approve_candidate(
        &path,
        &CandidateApproveRequest {
            id: submitted.candidate.id.clone(),
            embedding: None,
            embedding_model_id: None,
            dry_run: false,
        },
    )
    .expect("adjudicated capture candidate approves");
    assert_eq!(report.candidate.status, "approved");
    assert!(report.candidate.resulting_memory_id.is_some());

    cleanup_store(&path);
}

#[test]
fn approve_promotes_unadjudicated_capture_when_not_required() {
    // Default posture (MEMKEEPER_CAPTURE_REQUIRE_ADJUDICATION unset): a capture
    // candidate with no verdict still promotes (Degraded), so the gate is opt-in.
    let path = temp_store_path("candidate_capture_degraded");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = candidate_submit_request("fact: unadjudicated but allowed by default");
    request.source_type = Some("capture".to_string());
    let submitted = submit_candidate(&path, &request).expect("submit succeeds");

    let report = approve_candidate(
        &path,
        &CandidateApproveRequest {
            id: submitted.candidate.id.clone(),
            embedding: None,
            embedding_model_id: None,
            dry_run: false,
        },
    )
    .expect("unadjudicated capture approves when not required");
    assert_eq!(report.candidate.status, "approved");

    cleanup_store(&path);
}

#[test]
fn quarantine_candidate_transitions_pending_and_blocks_decided() {
    use crate::{quarantine_candidate, CandidateQuarantineRequest};
    let path = temp_store_path("candidate_quarantine");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = candidate_submit_request("fact: adjudicator flagged this as unsupported");
    request.source_type = Some("capture".to_string());
    let submitted = submit_candidate(&path, &request).expect("submit succeeds");

    let report = quarantine_candidate(
        &path,
        &CandidateQuarantineRequest {
            id: submitted.candidate.id.clone(),
            reason: Some("unsupported edge; no source quote".to_string()),
            dry_run: false,
        },
    )
    .expect("quarantine succeeds");
    assert_eq!(report.candidate.status, "quarantined");
    assert_eq!(
        report.candidate.decided_reason.as_deref(),
        Some("unsupported edge; no source quote")
    );

    // Quarantined is a terminal state -> a second decision is refused.
    assert!(matches!(
        quarantine_candidate(
            &path,
            &CandidateQuarantineRequest {
                id: submitted.candidate.id.clone(),
                reason: None,
                dry_run: false,
            },
        ),
        Err(Error::InvalidRequest { .. })
    ));

    // And a quarantined candidate is listable under its status.
    let listed = list_candidates(
        &path,
        &CandidateListRequest {
            status: Some("quarantined".to_string()),
            space: None,
            limit: 50,
            offset: 0,
        },
    )
    .expect("list quarantined");
    assert_eq!(listed.total, 1);

    cleanup_store(&path);
}

