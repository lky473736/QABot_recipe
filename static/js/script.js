/**
 * 레시피 챗봇 JavaScript
 */

// 전역 변수
let chatHistory = [];
let isTyping = false;

// DOM 로드 완료 시 실행
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// 앱 초기화
function initializeApp() {
    console.log('레시피 챗봇 앱 초기화 중...');
    
    // 서비스 워커 등록 (PWA 지원)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/static/js/sw.js')
            .then(registration => console.log('Service Worker 등록 성공'))
            .catch(error => console.log('Service Worker 등록 실패'));
    }
    
    // 챗봇 상태 확인
    checkChatbotStatus();
    
    // 이벤트 리스너 등록
    setupEventListeners();
    
    // 로컬 스토리지에서 채팅 기록 복원
    restoreChatHistory();
}

// 챗봇 상태 확인
function checkChatbotStatus() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            if (data.chatbot_loaded) {
                console.log(`챗봇 로드 완료 - 레시피: ${data.recipes_count}개, QA: ${data.qa_count}개`);
                showNotification('챗봇이 준비되었습니다!', 'success');
            } else {
                console.warn('챗봇 로드 실패');
                showNotification('챗봇 로드에 실패했습니다.', 'warning');
            }
        })
        .catch(error => {
            console.error('챗봇 상태 확인 실패:', error);
            showNotification('서버 연결에 실패했습니다.', 'error');
        });
}

// 이벤트 리스너 설정
function setupEventListeners() {
    // 키보드 이벤트
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // 윈도우 이벤트
    window.addEventListener('beforeunload', saveChatHistory);
    window.addEventListener('resize', handleWindowResize);
    
    // 터치 이벤트 (모바일 지원)
    if ('ontouchstart' in window) {
        document.addEventListener('touchstart', handleTouchStart, {passive: true});
        document.addEventListener('touchmove', handleTouchMove, {passive: true});
    }
}

// 키보드 단축키 처리
function handleKeyboardShortcuts(event) {
    // Ctrl + Enter: 메시지 전송
    if (event.ctrlKey && event.key === 'Enter') {
        const messageInput = document.getElementById('messageInput');
        if (messageInput && document.activeElement === messageInput) {
            sendMessage();
        }
    }
    
    // ESC: 입력 필드 초기화
    if (event.key === 'Escape') {
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.value = '';
            messageInput.blur();
        }
    }
    
    // Ctrl + L: 채팅 초기화
    if (event.ctrlKey && event.key === 'l') {
        event.preventDefault();
        if (confirm('채팅 내역을 모두 삭제하시겠습니까?')) {
            clearChat();
        }
    }
}

// 윈도우 리사이즈 처리
function handleWindowResize() {
    // 채팅 메시지 스크롤 조정
    const messagesContainer = document.getElementById('chatMessages');
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// 터치 이벤트 처리
let touchStartY = 0;
function handleTouchStart(event) {
    touchStartY = event.touches[0].clientY;
}

function handleTouchMove(event) {
    const touchY = event.touches[0].clientY;
    const diff = touchStartY - touchY;
    
    // 풀투리프레시 방지
    if (diff < 0 && window.scrollY === 0) {
        event.preventDefault();
    }
}

// 알림 표시
function showNotification(message, type = 'info', duration = 3000) {
    // 기존 알림 제거
    const existingToast = document.querySelector('.toast-notification');
    if (existingToast) {
        existingToast.remove();
    }
    
    // 새 알림 생성
    const toast = document.createElement('div');
    toast.className = `toast-notification alert alert-${getBootstrapAlertClass(type)} position-fixed`;
    toast.style.cssText = `
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        animation: slideInRight 0.3s ease;
    `;
    
    toast.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas ${getNotificationIcon(type)} me-2"></i>
            <span>${message}</span>
            <button type="button" class="btn-close ms-auto" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // 자동 제거
    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }
    }, duration);
}

// 알림 타입에 따른 Bootstrap 클래스 반환
function getBootstrapAlertClass(type) {
    const typeMap = {
        success: 'success',
        error: 'danger',
        warning: 'warning',
        info: 'info'
    };
    return typeMap[type] || 'info';
}

// 알림 타입에 따른 아이콘 반환
function getNotificationIcon(type) {
    const iconMap = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    return iconMap[type] || 'fa-info-circle';
}

// 채팅 기록 저장
function saveChatHistory() {
    try {
        localStorage.setItem('recipebot_chat_history', JSON.stringify(chatHistory));
    } catch (error) {
        console.warn('채팅 기록 저장 실패:', error);
    }
}

// 채팅 기록 복원
function restoreChatHistory() {
    try {
        const savedHistory = localStorage.getItem('recipebot_chat_history');
        if (savedHistory) {
            chatHistory = JSON.parse(savedHistory);
            
            // 채팅 페이지에서만 기록 복원
            if (window.location.pathname === '/chat') {
                restoreMessages();
            }
        }
    } catch (error) {
        console.warn('채팅 기록 복원 실패:', error);
        chatHistory = [];
    }
}

// 메시지 복원
function restoreMessages() {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer || chatHistory.length === 0) return;
    
    // 기존 메시지 제거 (환영 메시지 제외)
    const welcomeMessage = messagesContainer.querySelector('.message');
    messagesContainer.innerHTML = '';
    if (welcomeMessage) {
        messagesContainer.appendChild(welcomeMessage);
    }
    
    // 저장된 메시지 복원
    chatHistory.forEach(message => {
        addMessageToDOM(message.text, message.sender, message.timestamp);
    });
}

// DOM에 메시지 추가 (개선된 버전)
function addMessageToDOM(text, sender, timestamp = null) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const now = timestamp ? new Date(timestamp) : new Date();
    const timeString = now.toLocaleTimeString('ko-KR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    // 메시지 내용 처리 (마크다운 스타일 지원)
    const processedText = processMessageText(text);
    
    if (sender === 'bot') {
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-text">${processedText}</div>
                <div class="message-time">${timeString}</div>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${processedText}</div>
                <div class="message-time">${timeString}</div>
            </div>
            <div class="message-avatar">
                <i class="fas fa-user"></i>
            </div>
        `;
    }
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // 채팅 기록에 추가
    if (!timestamp) { // 새 메시지인 경우만
        chatHistory.push({
            text: text,
            sender: sender,
            timestamp: now.toISOString()
        });
        
        // 최대 100개 메시지만 저장
        if (chatHistory.length > 100) {
            chatHistory = chatHistory.slice(-100);
        }
    }
}

// 메시지 텍스트 처리 (간단한 마크다운 지원)
function processMessageText(text) {
    return text
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
}

// 타이핑 표시
function showTypingIndicator() {
    if (isTyping) return;
    
    isTyping = true;
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="message-text">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// 타이핑 표시 제거
function hideTypingIndicator() {
    isTyping = false;
    const typingIndicator = document.querySelector('.typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// 메시지 검색
function searchMessages(query) {
    const messages = document.querySelectorAll('.message-text');
    let results = [];
    
    messages.forEach((message, index) => {
        if (message.textContent.toLowerCase().includes(query.toLowerCase())) {
            results.push({
                index: index,
                element: message,
                text: message.textContent
            });
        }
    });
    
    return results;
}

// 메시지 하이라이트
function highlightMessage(messageElement) {
    messageElement.style.backgroundColor = '#fff3cd';
    setTimeout(() => {
        messageElement.style.backgroundColor = '';
    }, 2000);
}

// 채팅 내보내기
function exportChat() {
    const chatData = {
        timestamp: new Date().toISOString(),
        messages: chatHistory
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recipe_chat_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// 음성 인식 (Web Speech API)
function startVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        showNotification('음성 인식을 지원하지 않는 브라우저입니다.', 'warning');
        return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.lang = 'ko-KR';
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onstart = function() {
        showNotification('음성 인식을 시작합니다...', 'info');
    };
    
    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.value = transcript;
            showNotification('음성 인식 완료!', 'success');
        }
    };
    
    recognition.onerror = function(event) {
        showNotification('음성 인식 오류가 발생했습니다.', 'error');
    };
    
    recognition.start();
}

// CSS 애니메이션 추가
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .typing-dots {
        display: flex;
        gap: 3px;
        align-items: center;
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #007bff;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .typing-dots span:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);

// 전역 함수로 내보내기 (HTML에서 사용)
window.recipeBot = {
    showNotification,
    exportChat,
    startVoiceRecognition,
    searchMessages,
    highlightMessage
};