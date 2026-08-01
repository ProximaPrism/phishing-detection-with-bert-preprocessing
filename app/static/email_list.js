async function predict(email) {

    const response = await fetch("/predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(email)

    });

    const result = await response.json();

    const badge = document.getElementById("prediction");

    if (result.prediction === 1) {

        badge.className = "badge phishing";

        badge.textContent =
            `⚠️ Phishing (${(result.confidence * 100).toFixed(1)}%)`;

    } else {

        badge.className = "badge safe";

        badge.textContent =
            `✔ Legitimate (${(result.confidence * 100).toFixed(1)}%)`;

    }

}

async function openEmail(email, element) {

    document.querySelectorAll(".email")
        .forEach(e => e.classList.remove("active"));

    element.classList.add("active");

    document.getElementById("subject").textContent =
        email.subject;

    document.getElementById("from").textContent =
        `${email.sender_display_name ?? ""} <${email.sender_email}>`;

    document.getElementById("date").textContent =
        email.sent_datetime;

    document.getElementById("body").textContent =
        email.body;

    await predict(email);

}

async function loadInbox() {

    const response = await fetch("/emails");
    const emails = await response.json();

    const list = document.getElementById("emailList");

    list.innerHTML = "";

    emails.forEach(email => {

        const item = document.createElement("div");

        item.className = "email";

        item.innerHTML = `
            <h3>${email.subject}</h3>
            <p>${email.sender_display_name ?? email.sender_email}</p>
        `;

        item.onclick = () => openEmail(email, item);

        list.appendChild(item);

    });
}

window.onload = loadInbox;