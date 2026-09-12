const uploadForm = document.getElementById("upload-form");
const resumeInput = document.getElementById("resume-input");
const uploadResults = document.getElementById("upload-results");

const jdForm = document.getElementById("jd-form");
const jdText = document.getElementById("jd-text");
const jdFields = document.getElementById("jd-fields");

const candidatesTable = document.getElementById("candidates-table");
const clearCandidatesBtn = document.getElementById("clear-candidates-btn");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatWindow = document.getElementById("chat-window");

function renderChips(items, cssClass) {
  if (!items || items.length === 0) return "<span class=\"hint\">none</span>";
  return items.map((item) => `<span class="chip ${cssClass || ""}">${item}</span>`).join("");
}

function renderCandidates(candidates) {
  if (!candidates || candidates.length === 0) {
    candidatesTable.innerHTML = '<p class="hint">No candidates ranked yet.</p>';
    return;
  }

  const rows = candidates
    .map((c) => {
      const score = c.score || {};
      return `
        <tr>
          <td>${c.filename}</td>
          <td class="score-cell">${score.overall ?? "-"}</td>
          <td>${score.semantic ?? "-"}</td>
          <td>${score.skills ?? "-"}</td>
          <td>${score.experience ?? "-"}</td>
          <td>${score.education ?? "-"}</td>
          <td>${renderChips(score.matched_skills, "matched")}</td>
          <td>${renderChips(score.missing_skills, "missing")}</td>
        </tr>`;
    })
    .join("");

  candidatesTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Candidate</th><th>Overall</th><th>Semantic</th><th>Skills</th>
          <th>Experience</th><th>Education</th><th>Matched</th><th>Missing</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!resumeInput.files.length) return;

  const formData = new FormData();
  for (const file of resumeInput.files) formData.append("resumes", file);

  uploadResults.innerHTML = '<li class="hint">Processing...</li>';
  const submitBtn = uploadForm.querySelector("button");
  submitBtn.disabled = true;

  try {
    const response = await fetch("/api/upload_resumes", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      uploadResults.innerHTML = `<li class="error">${data.error || "Upload failed."}</li>`;
      return;
    }

    uploadResults.innerHTML = data.results
      .map((r) => {
        if (r.status === "success") {
          return `<li class="success">${r.filename} — processed (${r.extraction_method})</li>`;
        }
        return `<li class="error">${r.filename} — ${r.message}</li>`;
      })
      .join("");

    renderCandidates(data.candidates);
  } catch (err) {
    uploadResults.innerHTML = `<li class="error">${err}</li>`;
  } finally {
    submitBtn.disabled = false;
  }
});

jdForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = jdText.value.trim();
  if (!text) return;

  const submitBtn = jdForm.querySelector("button");
  submitBtn.disabled = true;
  jdFields.innerHTML = '<p class="hint">Analyzing...</p>';

  try {
    const response = await fetch("/api/job_description", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();

    if (!response.ok) {
      jdFields.innerHTML = `<p class="hint">${data.error || "Failed to analyze."}</p>`;
      return;
    }

    const f = data.fields;
    jdFields.innerHTML = `
      <p><strong>Required skills:</strong> ${renderChips(f.skills)}</p>
      <p><strong>Min. experience:</strong> ${f.experience_years ?? "not specified"} years</p>
      <p><strong>Education:</strong> ${renderChips(f.education)}</p>
      <p class="hint">Extracted via: ${f.source}</p>`;

    renderCandidates(data.candidates);
  } catch (err) {
    jdFields.innerHTML = `<p class="hint">${err}</p>`;
  } finally {
    submitBtn.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;

  appendChatMessage(query, "user");
  chatInput.value = "";

  const thinkingMsg = appendChatMessage("Thinking...", "bot");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await response.json();
    thinkingMsg.textContent = response.ok ? data.answer : data.error || "Something went wrong.";
  } catch (err) {
    thinkingMsg.textContent = String(err);
  }
});

function appendChatMessage(text, role) {
  const msg = document.createElement("div");
  msg.className = `chat-msg ${role}`;
  msg.textContent = text;
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return msg;
}

async function refreshCandidates() {
  try {
    const response = await fetch("/api/candidates");
    const data = await response.json();
    renderCandidates(data.candidates);
  } catch (err) {
    // Non-fatal on initial load.
  }
}

clearCandidatesBtn.addEventListener("click", async () => {
  if (!confirm("Remove all uploaded candidates? This can't be undone.")) return;

  clearCandidatesBtn.disabled = true;
  try {
    const response = await fetch("/api/candidates", { method: "DELETE" });
    const data = await response.json();
    renderCandidates(data.candidates);
    uploadResults.innerHTML = "";
    resumeInput.value = "";
  } catch (err) {
    alert(`Failed to clear candidates: ${err}`);
  } finally {
    clearCandidatesBtn.disabled = false;
  }
});

refreshCandidates();
