let offset = 0;
let limit = 10;
let loading = false;
let finishedLoading = false;
let redirecting = false;


function redirectToLogin() {
    if (redirecting) {
        return;
    }

    redirecting = true;
    finishedLoading = true;
    loading = false;

    window.location.replace("/");
}

async function loadEmails() {
    if (loading || finishedLoading || redirecting) {
        return;
    }
    loading = true;

    try {
        const response = await fetch(
            `/emails?offset=${offset}`,
            {
                credentials: "include"
            }
        );

        if (response.status === 401) {
            redirectToLogin();
            return;
        }
        if (!response.ok) {
            throw new Error(
                "Unable to load emails."
            );
        }

        const emails = await response.json();
        const list =
            document.getElementById(
                "emailList"
            );

        if (!emails.length) {
            finishedLoading = true;
            return;
        }
        emails.forEach(email => {
            createEmailItem(
                email,
                list
            );
        });

        offset += emails.length;

        if (emails.length < limit) {
            finishedLoading = true;
        }
    } catch (error) {
        console.error(error);
    } finally {
        loading = false;
    }
}

function createEmailItem(email, list) {
    const item =
        document.createElement("div");

    item.className = "email";
    item.innerHTML = `
        <h3>
            ${escapeHTML(email.subject)}
        </h3>
        <p>
            ${
        escapeHTML(
            email.sender_display_name ||
            email.sender_email
        )
    }
        </p>
    `;

    item.onclick = () => {
        openEmail(
            email,
            item
        );
    };
    list.appendChild(item);
}


async function openEmail(email, element) {
    document
        .querySelectorAll(".email")
        .forEach(item => {
            item.classList.remove(
                "active"
            );
        });
    element.classList.add("active");

    document.getElementById("subject").textContent = email.subject;
    document.getElementById("from").textContent = `${email.sender_display_name ?? ""}<${email.sender_email}>`;
    document.getElementById("date").textContent = email.sent_datetime;
    document.getElementById("body").textContent = email.body;

    await predictEmail(email);
}


async function predictEmail(email) {
    const badge = document.getElementById("prediction");

    badge.className = "badge";
    badge.textContent = "Analysing...";

    try {
        const response = await fetch(
            "/predict",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(email)
            }
        );

        if (!response.ok) {
            throw new Error(
                "Prediction failed."
            );
        }

        const result = await response.json();

        const confidence = (
            result.confidence * 100
        ).toFixed(1);

        let explanation = "";

        // Only show explanations for phishing emails
        if (
            result.prediction === 1 &&
            result.analysis
        ) {
            const triggerWords =
                result.analysis.trigger_words || [];

            const reasons =
                result.analysis.reasons || [];

            if (
                triggerWords.length > 0 ||
                reasons.length > 0
            ) {
                explanation = `
                    <div class="triggers">
                        ${
                    triggerWords.length > 0
                        ? `<strong>Triggers:</strong>
                                   ${escapeHTML(
                            triggerWords.join(", ")
                        )}`
                        : ""
                }

                        ${
                    reasons.length > 0
                        ? `<br><strong>Reasons:</strong>
                                   ${escapeHTML(
                            reasons.join("\n\n")
                        )}`
                        : ""
                }
                    </div>
                `;
            }
        }

        if (result.prediction === 1) {
            badge.className = "badge phishing";
            badge.innerHTML =
                `⚠️ Phishing (${confidence}%)${explanation}`;
        } else {
            badge.className = "badge safe";
            badge.innerHTML =
                `✔ Legitimate (${confidence}%)`;
        }

    } catch (error) {
        console.error(error);

        badge.className = "badge";
        badge.textContent = "Unable to analyse";
    }
}


function setupLazyLoading() {
    const list =
        document.getElementById(
            "emailList"
        );

    list.addEventListener(
        "scroll",
        () => {
            const nearBottom =
                list.scrollTop +
                list.clientHeight >=
                list.scrollHeight - 50;

            if (nearBottom) {
                loadEmails();
            }
        }
    );
}


function escapeHTML(value) {
    if (!value) {
        return "";
    }

    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

window.onload = async () => {
    const list = document.getElementById("emailList");

    setupLazyLoading();
    await loadEmails();

    // keep loading until scrolling becomes possible
    while (
        !redirecting &&
        list.scrollHeight <= list.clientHeight &&
        !finishedLoading
        ) {
        await loadEmails();
    }

};