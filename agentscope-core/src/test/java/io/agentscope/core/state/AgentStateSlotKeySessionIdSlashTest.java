/*
 * Copyright 2024-2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.agentscope.core.state;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.agentscope.core.ReActAgent;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

/**
 * Regression tests for <a
 * href="https://github.com/agentscope-ai/agentscope-java/issues/2475">#2475</a>.
 *
 * <p>Bug: {@code ReActAgent} builds an internal slot key as {@code userId + "/" + sessionId}, and
 * end-of-call persistence used to re-derive the pair by splitting that key at the <em>last</em>
 * {@code '/'} ({@code SlotRef.parse}). When the sessionId itself contains {@code '/'} the split
 * produced a corrupted pair, so {@code AgentState} was written under the wrong keys and any later
 * read with the original {@code (userId, sessionId)} returned empty/fresh state — silently breaking
 * conversation-history replay.
 *
 * <p>This is not an exotic input: DingTalk {@code openConversationId} values are standard Base64 and
 * legitimately contain {@code '/'}.
 *
 * <p>Fixed by storing {@code userId} / {@code sessionId} directly on {@code CallExecution} instead of
 * round-tripping through the concatenated slot key. These tests pin the behavior end-to-end through a
 * real {@link ReActAgent} call against an {@link InMemoryAgentStateStore}.
 */
@DisplayName("#2475: AgentState slot integrity when sessionId contains '/'")
class AgentStateSlotKeySessionIdSlashTest {

    private static final Duration TIMEOUT = Duration.ofSeconds(15);
    private static final String STATE_KEY = "agent_state";

    /** A real DingTalk-style openConversationId: Base64, contains a single '/'. */
    private static final String DINGTALK_SESSION_ID =
            "cidLcMOzfRs62haqnq1NO8SxeYdmc/lO8bWyidOisoKAzM=";

    private static final String USER_ID = "u1";

    /** Minimal model: answers with plain text so the ReAct loop terminates after one turn. */
    private static final class TextOnlyModel extends ChatModelBase {

        @Override
        public String getModelName() {
            return "text-only";
        }

        @Override
        protected Flux<ChatResponse> doStream(
                List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            return Flux.just(
                    ChatResponse.builder()
                            .content(List.<ContentBlock>of(TextBlock.builder().text("ok").build()))
                            .build());
        }
    }

    private static Msg userMsg(String text) {
        return Msg.builder()
                .name("user")
                .role(MsgRole.USER)
                .content(List.<ContentBlock>of(TextBlock.builder().text(text).build()))
                .build();
    }

    private static ReActAgent buildAgent(AgentStateStore store) {
        return ReActAgent.builder()
                .name("asst")
                .sysPrompt("You are concise.")
                .model(new TextOnlyModel())
                .stateStore(store)
                .build();
    }

    @Test
    @DisplayName("sessionId with '/': state is persisted under the original (userId, sessionId)")
    void sessionIdWithSlash_statePersistedUnderOriginalSlot() {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = buildAgent(store);

        Msg reply =
                agent.call(
                                List.of(userMsg("hi")),
                                RuntimeContext.builder()
                                        .userId(USER_ID)
                                        .sessionId(DINGTALK_SESSION_ID)
                                        .build())
                        .block(TIMEOUT);
        assertNotNull(reply, "agent call returned no reply");

        Optional<AgentState> saved =
                store.get(USER_ID, DINGTALK_SESSION_ID, STATE_KEY, AgentState.class);
        assertTrue(
                saved.isPresent(),
                "#2475 regression: no AgentState under the original (userId, sessionId) — the slot"
                        + " key was split at the '/' inside the sessionId");
    }

    @Test
    @DisplayName("sessionId with '/': nothing is written to the corrupted split pair")
    void sessionIdWithSlash_noStateUnderCorruptedSlot() {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = buildAgent(store);

        agent.call(
                        List.of(userMsg("hi")),
                        RuntimeContext.builder()
                                .userId(USER_ID)
                                .sessionId(DINGTALK_SESSION_ID)
                                .build())
                .block(TIMEOUT);

        // Reproduce exactly what the old SlotRef.parse() would have derived: split
        // "u1/cidLcMOzfRs62haqnq1NO8SxeYdmc/lO8bWyidOisoKAzM=" at the LAST '/'.
        String slotKey = USER_ID + "/" + DINGTALK_SESSION_ID;
        int lastSlash = slotKey.lastIndexOf('/');
        String corruptedUserId = slotKey.substring(0, lastSlash);
        String corruptedSessionId = slotKey.substring(lastSlash + 1);

        // Guard: the fixture must actually exercise the corruption path.
        assertEquals("u1/cidLcMOzfRs62haqnq1NO8SxeYdmc", corruptedUserId);
        assertEquals("lO8bWyidOisoKAzM=", corruptedSessionId);

        Optional<AgentState> leaked =
                store.get(corruptedUserId, corruptedSessionId, STATE_KEY, AgentState.class);
        assertTrue(
                leaked.isEmpty(),
                "#2475 regression: AgentState leaked into the corrupted slot ("
                        + corruptedUserId
                        + ", "
                        + corruptedSessionId
                        + ") — state store accumulates rows under bogus userId prefixes");
    }

    @Test
    @DisplayName("sessionId with '/': second call replays the first call's state")
    void sessionIdWithSlash_stateReplayedAcrossCalls() {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = buildAgent(store);

        RuntimeContext ctx =
                RuntimeContext.builder().userId(USER_ID).sessionId(DINGTALK_SESSION_ID).build();

        agent.call(List.of(userMsg("first")), ctx).block(TIMEOUT);
        Optional<AgentState> afterFirst =
                store.get(USER_ID, DINGTALK_SESSION_ID, STATE_KEY, AgentState.class);
        assertTrue(afterFirst.isPresent(), "state missing after first call");

        agent.call(List.of(userMsg("second")), ctx).block(TIMEOUT);
        Optional<AgentState> afterSecond =
                store.get(USER_ID, DINGTALK_SESSION_ID, STATE_KEY, AgentState.class);
        assertTrue(
                afterSecond.isPresent(),
                "#2475 regression: state under the original slot disappeared after the second"
                        + " call");
    }

    @Test
    @DisplayName("anonymous user + sessionId with '/': state still lands on the original slot")
    void anonymousUserSessionIdWithSlash_statePersistedUnderOriginalSlot() {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = buildAgent(store);

        // No userId: slotKey becomes "__anon__/<sessionId>", so the '/' inside the sessionId is
        // still the last one and the old parse() corrupted this case too.
        agent.call(
                        List.of(userMsg("hi")),
                        RuntimeContext.builder().sessionId(DINGTALK_SESSION_ID).build())
                .block(TIMEOUT);

        Optional<AgentState> saved =
                store.get(null, DINGTALK_SESSION_ID, STATE_KEY, AgentState.class);
        assertTrue(
                saved.isPresent(),
                "#2475 regression: anonymous state not stored under the original sessionId");
    }

    @Test
    @DisplayName("sessionId with multiple '/': state still lands on the original slot")
    void sessionIdWithMultipleSlashes_statePersistedUnderOriginalSlot() {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = buildAgent(store);

        String multiSlash = "a/b/c/d";
        agent.call(
                        List.of(userMsg("hi")),
                        RuntimeContext.builder().userId(USER_ID).sessionId(multiSlash).build())
                .block(TIMEOUT);

        Optional<AgentState> saved = store.get(USER_ID, multiSlash, STATE_KEY, AgentState.class);
        assertTrue(
                saved.isPresent(),
                "#2475 regression: multi-slash sessionId not stored under the original slot");
    }

    @Test
    @DisplayName("sessionId without '/': unchanged behavior (no regression from the fix)")
    void sessionIdWithoutSlash_unchangedBehavior() {
        InMemoryAgentStateStore store = new InMemoryAgentStateStore();
        ReActAgent agent = buildAgent(store);

        String plainSession = "session-1";
        agent.call(
                        List.of(userMsg("hi")),
                        RuntimeContext.builder().userId(USER_ID).sessionId(plainSession).build())
                .block(TIMEOUT);

        Optional<AgentState> saved = store.get(USER_ID, plainSession, STATE_KEY, AgentState.class);
        assertTrue(saved.isPresent(), "plain sessionId must keep working after the fix");
    }
}
