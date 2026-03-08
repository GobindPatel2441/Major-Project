const STORAGE_KEY = "mika_chats_v1";
const THEME_KEY = "mika_theme_v1";
let chats = [];
let activeChatId = null;
let isSending = false;
let typingNode = null;
let apiBase = null;

const API_BASE_CANDIDATES = [
    "http://127.0.0.1:5000",
    "http://localhost:5000"
];

window.onload = () => {
    wireUi();
    loadChats();
    renderChatList();
    if (!activeChatId) {
        createNewChat();
    } else {
        setActiveChat(activeChatId);
    }

    checkServerStatus();
    setInterval(checkServerStatus, 5000);
};

function wireUi() {
    const newChatBtn = document.getElementById("new-chat");
    const toggleBtn = document.getElementById("toggle-sidebar");
    const themeSwitch = document.getElementById("theme-switch");

    newChatBtn.addEventListener("click", createNewChat);
    toggleBtn.addEventListener("click", () => {
        document.body.classList.toggle("sidebar-collapsed");
    });

    const savedTheme = localStorage.getItem(THEME_KEY) || "dark";
    applyTheme(savedTheme, themeSwitch);

    themeSwitch.addEventListener("change", () => {
        const nextTheme = themeSwitch.checked ? "dark" : "light";
        applyTheme(nextTheme, themeSwitch);
        localStorage.setItem(THEME_KEY, nextTheme);
    });
}

function applyTheme(theme, themeSwitch) {
    const isDark = theme === "dark";
    document.body.classList.toggle("theme-light", !isDark);
    if (themeSwitch) {
        themeSwitch.checked = isDark;
    }
}

function defaultChat() {
    const id = `chat_${Date.now()}`;
    return {
        id,
        title: "Mike",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [
            {
                role: "bot",
                text: "Hi! I'm Mika. How are you feeling today?",
                ts: Date.now()
            }
        ]
    };
}

function loadChats() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            chats = Array.isArray(parsed.chats) ? parsed.chats : [];
            activeChatId = parsed.activeChatId || null;
        }
    } catch (err) {
        chats = [];
        activeChatId = null;
    }

    // Migrate old titles
    let migrated = false;
    chats = chats.map((chat) => {
        if (chat.title === "Chat with Mika") {
            migrated = true;
            return { ...chat, title: "Mike" };
        }
        return chat;
    });
    if (migrated) {
        saveChats();
    }
}

function saveChats() {
    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ chats, activeChatId })
    );
}

function createNewChat() {
    const chat = defaultChat();
    chats.unshift(chat);
    activeChatId = chat.id;
    saveChats();
    renderChatList();
    setActiveChat(chat.id);
}

function deleteChat(chatId) {
    const chat = chats.find((item) => item.id === chatId);
    if (!chat) return;

    const ok = window.confirm(`Delete "${chat.title}"? This cannot be undone.`);
    if (!ok) return;

    chats = chats.filter((item) => item.id !== chatId);
    if (activeChatId === chatId) {
        activeChatId = chats.length ? chats[0].id : null;
    }
    saveChats();
    renderChatList();
    if (activeChatId) {
        setActiveChat(activeChatId);
    } else {
        createNewChat();
    }
}

function setActiveChat(chatId) {
    activeChatId = chatId;
    saveChats();
    renderChatList();
    renderMessages();
    updateHeader();
}

function updateHeader() {
    const title = document.getElementById("chat-title");
    const chat = chats.find((item) => item.id === activeChatId);
    title.textContent = chat ? chat.title : "Mike";
}

async function checkServerStatus() {
    const statusEl = document.querySelector(".status");
    const sidebarDot = document.querySelector(".brand-dot");
    if (!statusEl) return;

    try {
        const base = await resolveApiBase();
        const res = await fetch(`${base}/status`);
        const data = await res.json();
        const isOnline = data && data.status === "online";

        if (isOnline) {
            statusEl.classList.remove("offline");
            statusEl.textContent = "Active now";
            if (sidebarDot) sidebarDot.style.background = "#22c55e";
        } else {
            statusEl.classList.add("offline");
            statusEl.textContent = "Offline";
            if (sidebarDot) sidebarDot.style.background = "#ef4444";
        }
    } catch (err) {
        console.error("Failed to fetch server status", err);
        statusEl.classList.add("offline");
        statusEl.textContent = "Offline";
        if (sidebarDot) sidebarDot.style.background = "#ef4444";
    }
}

async function resolveApiBase() {
    if (apiBase) return apiBase;

    for (const candidate of API_BASE_CANDIDATES) {
        const url = `${candidate}/status`;
        try {
            const res = await fetch(url);
            if (!res.ok) continue;
            const data = await res.json();
            if (data && typeof data.status === "string") {
                apiBase = candidate;
                return apiBase;
            }
        } catch (_err) {
            // Try next candidate.
        }
    }
    throw new Error("No reachable backend API base URL.");
}

function renderChatList() {
    const list = document.getElementById("chat-list");
    list.innerHTML = "";

    const sorted = [...chats].sort((a, b) => b.updatedAt - a.updatedAt);
    sorted.forEach((chat) => {
        const item = document.createElement("div");
        item.className = `chat-item${chat.id === activeChatId ? " active" : ""}`;

        const img = document.createElement("img");
        img.src = "https://i.pinimg.com/564x/8d/ff/49/8dff49985d0d8afa53751d9ba8907aed.jpg";
        img.alt = "Avatar";

        const content = document.createElement("div");
        content.className = "chat-item-content";

        const title = document.createElement("h4");
        title.textContent = chat.title;

        const preview = document.createElement("p");
        preview.textContent = getPreview(chat);

        content.appendChild(title);
        content.appendChild(preview);

        const time = document.createElement("span");
        time.className = "time";
        time.textContent = formatTime(chat.updatedAt);

        const del = document.createElement("button");
        del.className = "delete-btn";
        del.type = "button";
        del.title = "Delete chat";
        del.textContent = "✕";
        del.addEventListener("click", (event) => {
            event.stopPropagation();
            deleteChat(chat.id);
        });

        item.appendChild(img);
        item.appendChild(content);
        item.appendChild(time);
        item.appendChild(del);

        item.addEventListener("click", () => setActiveChat(chat.id));
        list.appendChild(item);
    });
}

function getPreview(chat) {
    const last = chat.messages[chat.messages.length - 1];
    if (!last) return "No messages yet";
    if (last.text.length <= 36) return last.text;
    return `${last.text.slice(0, 36)}...`;
}

function formatTime(ts) {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;

    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    const isYesterday =
        date.getFullYear() === yesterday.getFullYear() &&
        date.getMonth() === yesterday.getMonth() &&
        date.getDate() === yesterday.getDate();
    if (isYesterday) return "Yesterday";

    if (diffDays < 7) {
        return date.toLocaleDateString(undefined, { weekday: "short" });
    }

    if (date.getFullYear() === now.getFullYear()) {
        return date.toLocaleDateString(undefined, {
            month: "short",
            day: "numeric"
        });
    }

    return date.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric"
    });
}

function renderMessages() {
    const chat = document.getElementById("chat");
    chat.innerHTML = "";

    const active = chats.find((item) => item.id === activeChatId);
    if (!active) return;

    active.messages.forEach((msg) => {
        const bubble = document.createElement("div");
        bubble.className = `message ${msg.role === "user" ? "user" : "bot"}`;
        bubble.textContent = msg.text;
        chat.appendChild(bubble);
    });

    chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
}

function showTypingIndicator() {
    const chat = document.getElementById("chat");
    if (!chat) return;

    // Remove any existing indicator before adding a new one
    hideTypingIndicator();

    typingNode = document.createElement("div");
    typingNode.className = "typing-indicator";
    typingNode.textContent = "Mika is typing...";
    chat.appendChild(typingNode);
    chat.scrollTop = chat.scrollHeight;
}

function hideTypingIndicator() {
    if (typingNode && typingNode.parentNode) {
        typingNode.parentNode.removeChild(typingNode);
    }
    typingNode = null;
}

function setSendingState(sending) {
    const sendBtn = document.getElementById("send-btn");
    if (sendBtn) {
        sendBtn.disabled = sending;
        sendBtn.classList.toggle("is-loading", sending);
    }
    isSending = sending;
}

async function send() {
    const input = document.getElementById("input");
    const chat = document.getElementById("chat");
    const text = input.value.trim();

    if (!text || isSending) return;
    if (!activeChatId) createNewChat();

    const active = chats.find((item) => item.id === activeChatId);
    if (!active) return;

    setSendingState(true);

    active.messages.push({
        role: "user",
        text,
        ts: Date.now()
    });
    active.updatedAt = Date.now();
    saveChats();
    renderMessages();
    renderChatList();

    input.value = "";
    chat.scrollTo({ top: chat.scrollHeight, behavior: "smooth" });
    showTypingIndicator();

    try {
        // Backend call
        const base = await resolveApiBase();
        const history = active.messages.slice(-8).map((msg) => ({
            role: msg.role,
            text: msg.text
        }));
        const res = await fetch(`${base}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text, history })
        });

        if (!res.ok) {
            throw new Error(`Backend returned HTTP ${res.status}`);
        }
        const data = await res.json();

        active.messages.push({
            role: "bot",
            text: data.response,
            ts: Date.now()
        });
        active.updatedAt = Date.now();
        saveChats();
        renderMessages();
        renderChatList();
    } catch (err) {
        console.error("Failed to reach Mika backend", err);
        active.messages.push({
            role: "bot",
            text: "Sorry, backend is unreachable. Run start_project.bat, wait 10 seconds, then refresh this page.",
            ts: Date.now()
        });
        active.updatedAt = Date.now();
        saveChats();
        renderMessages();
        renderChatList();
    } finally {
        hideTypingIndicator();
        setSendingState(false);
    }
}
